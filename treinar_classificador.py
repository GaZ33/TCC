"""
Treina e avalia o classificador de letras do alfabeto manual (SVM; Random Forest como alternativa).

Uso (de dentro da pasta do projeto):
  .venv\\Scripts\\python treinar_classificador.py

Etapas:
  1. Lê os CSVs do coletor, um dataset por pessoa (DATASETS). Frames sem mão são descartados.
  2. Avaliação "pessoa de fora": para cada pessoa, treina com as outras e testa nela.
     É o número do TCC: o acerto com alguém que o modelo nunca viu.
  3. Treina o modelo final com todas as pessoas e salva em modelos/ (usado por classificador.py).
  4. Salva em resultados/ o relatório, as tabelas (CSV) e as figuras (PNG) da avaliação.

Por que não sortear frames para teste? Os 90 frames de uma tomada são quase iguais entre si.
Sorteando, o teste vira cópia do treino e o acerto sobe muito (99% com o Random Forest,
84% com a SVM), mas é ilusório.
Esse número aparece no relatório só para comparação.
"""

import datetime
import glob
import os
import re
import time

import joblib
import matplotlib
matplotlib.use("Agg")  # só gera arquivos, não abre janela
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from matplotlib.colors import LinearSegmentedColormap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

import caracteristicas_mao

# =====================================================================
# CONFIGURAÇÕES
# =====================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Um dataset por pessoa: nome -> pasta com os CSVs do coletor
DATASETS = {
    "Gabriel": os.path.join(SCRIPT_DIR, "novos_codigos"),
    "Fernando": os.path.join(SCRIPT_DIR, "tabelas", "DataSet - Fernando"),
    "Giovanni": os.path.join(SCRIPT_DIR, "tabelas", "DataSet - Giovanni"),
}

# Quais classes entram: "letras", "numeros" ou "todas".
# Letras e números juntos se confundem muito (2 e V, 6 e W, 9 e T, 3 e M têm a
# mesma forma de mão), então o padrão é treinar só as letras.
CLASSES = "letras"

# Modelo do sistema: "svm" (adotado depois de comparar_modelos.py) ou "random_forest"
MODELO = "svm"
# SVM com kernel RBF: mediana dos valores escolhidos nas 3 rodadas de comparar_modelos.py
SVM_C = 0.3
SVM_GAMMA = 0.003
N_ARVORES = 300     # Random Forest
SEMENTE = 42
PASTA_MODELOS = os.path.join(SCRIPT_DIR, "modelos")
PASTA_RESULTADOS = os.path.join(SCRIPT_DIR, "resultados")

# Nome dos arquivos do coletor: "letra a direita.csv", "letra 1a direito.csv",
# "letra n_direito.csv", "letra j esquerda_2.csv"... A mão vem do nome do arquivo.
PADRAO_NOME = re.compile(r"^letra (\w)\w*?[ _](direit|esquerd)[ao](_\d+)?\.csv$")

COLUNAS_XYZ = [f"lm_{i}_{eixo}" for i in range(21) for eixo in ("x", "y", "z")]

# Cores das figuras: escala de um só azul (claro = pouco, escuro = muito) e tinta neutra
AZUIS = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
COR_BARRA = "#2a78d6"
TINTA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
TINTA_FRACA = "#898781"
COR_GRADE = "#e1e0d9"


def num(valor, casas=2):
    """0.6 -> '0,60' (vírgula decimal, como no texto do TCC)."""
    return f"{valor:.{casas}f}".replace(".", ",")


def pct(valor, casas=1):
    """0.636 -> '63,6%'."""
    return num(valor * 100, casas) + "%"


# =====================================================================
# 1. LEITURA DOS DADOS
# =====================================================================
def ler_dataset(nome, pasta):
    """Lê os CSVs de uma pessoa.

    Retorna (dados, info): 'dados' tem só os frames COM mão, com as colunas x/y/z dos
    landmarks, 'label', 'esquerda' (mão esquerda) e 'tomada'; 'info' tem as contagens.
    Arquivos fora do padrão, sem as colunas ou com label diferente do nome são ignorados.
    """
    arquivos = sorted(glob.glob(os.path.join(pasta, "*.csv")))
    if not arquivos:
        raise SystemExit(f"[ERRO] Nenhum CSV em {pasta}")

    tomadas = []
    frames_sem_mao = 0
    for caminho in arquivos:
        arquivo = os.path.basename(caminho)
        casamento = PADRAO_NOME.match(arquivo)
        if not casamento:
            print(f"[AVISO] {nome}: '{arquivo}' fora do padrão 'letra <x> <mão>.csv', ignorado.")
            continue

        df = pd.read_csv(caminho, dtype={"label": str})
        if not set(COLUNAS_XYZ + ["hand_confidence", "label"]) <= set(df.columns):
            print(f"[AVISO] {nome}: '{arquivo}' não tem as colunas do coletor, ignorado.")
            continue

        letra = casamento.group(1).lower()
        labels = df["label"].str.lower().unique()
        if list(labels) != [letra]:
            print(f"[AVISO] {nome}: '{arquivo}' tem label {list(labels)}, mas o nome diz '{letra}'; ignorado.")
            continue

        # Frames sem mão têm landmarks zerados e confiança 0: não ensinam nada
        com_mao = df["hand_confidence"] > 0
        frames_sem_mao += int((~com_mao).sum())
        tomadas.append(df.loc[com_mao, COLUNAS_XYZ].assign(
            label=letra, esquerda=casamento.group(2) == "esquerd", tomada=f"{nome}/{arquivo}"))

    dados = pd.concat(tomadas, ignore_index=True)
    info = {"tomadas": len(tomadas), "frames": len(dados), "frames_sem_mao": frames_sem_mao}
    return dados, info


def filtrar_classes(dados):
    letra = dados["label"].str.isalpha()
    if CLASSES == "letras":
        return dados[letra]
    if CLASSES == "numeros":
        return dados[~letra]
    return dados


# =====================================================================
# 2. MODELO
# =====================================================================
def preparar(dados):
    """DataFrame de frames -> (X, y) para o Random Forest."""
    pontos = dados[COLUNAS_XYZ].to_numpy().reshape(-1, 21, 3)
    X = caracteristicas_mao.extrair(pontos, dados["esquerda"].to_numpy())
    y = np.asarray(dados["label"], dtype="U")
    return X, y


def criar_modelo():
    if MODELO == "svm":
        # Características padronizadas (o kernel RBF depende da escala) e probabilidades
        # calibradas, porque o filtro usa a confiança da letra. (O SVC(probability=True)
        # foi descontinuado no scikit-learn 1.9.)
        return make_pipeline(StandardScaler(), CalibratedClassifierCV(
            SVC(kernel="rbf", C=SVM_C, gamma=SVM_GAMMA), ensemble=False))
    return RandomForestClassifier(n_estimators=N_ARVORES, n_jobs=-1, random_state=SEMENTE)


def descricao_modelo():
    if MODELO == "svm":
        return f"SVM com kernel RBF (C = {num(SVM_C, 1)}, gamma = {num(SVM_GAMMA, 3)}), probabilidades calibradas"
    return f"Random Forest com {N_ARVORES} árvores (semente {SEMENTE})"


def treinar(X, y):
    # Pesos por amostra compensam classes com mais tomadas que outras.
    # (O class_weight="balanced" do scikit-learn 1.9.1 quebra com labels como '1', '2';
    #  por isso os pesos vão no fit.)
    pesos = compute_sample_weight("balanced", y)
    modelo = criar_modelo()
    if isinstance(modelo, Pipeline):
        modelo.fit(X, y, **{f"{modelo.steps[-1][0]}__sample_weight": pesos})
    else:
        modelo.fit(X, y, sample_weight=pesos)
    return modelo


def latencia_por_quadro_ms(modelo, X, repeticoes=200):
    """Tempo para classificar UM quadro com probabilidades, como no tempo real (em ms)."""
    if hasattr(modelo, "n_jobs"):
        modelo.n_jobs = 1
    modelo.predict_proba(X[:1])
    inicio = time.perf_counter()
    for i in range(repeticoes):
        modelo.predict_proba(X[i % len(X):i % len(X) + 1])
    return (time.perf_counter() - inicio) / repeticoes * 1000


# =====================================================================
# 3. AVALIAÇÃO
# =====================================================================
def avaliar_pessoa_de_fora(datasets):
    """Para cada pessoa: treina com as outras e prevê os frames dela.

    Retorna (previsoes, fora_da_conta): 'previsoes' tem uma linha por frame testado
    (pessoa, tomada, real, previsto); 'fora_da_conta' lista, por pessoa, as classes que
    não existem no treino (não há como acertá-las, então ficam fora da conta).
    """
    partes = []
    fora_da_conta = {}
    for teste, dados_teste in datasets.items():
        treino = pd.concat([d for nome, d in datasets.items() if nome != teste], ignore_index=True)
        modelo = treinar(*preparar(treino))

        sem_treino = sorted(set(dados_teste["label"]) - set(modelo.classes_))
        fora_da_conta[teste] = sem_treino
        dados_teste = dados_teste[~dados_teste["label"].isin(sem_treino)]

        X, y = preparar(dados_teste)
        partes.append(pd.DataFrame({"pessoa": teste, "tomada": dados_teste["tomada"].to_numpy(),
                                    "real": y, "previsto": modelo.predict(X)}))
    return pd.concat(partes, ignore_index=True), fora_da_conta


def votar_por_tomada(previsoes):
    """Uma linha por tomada: a classe mais prevista entre os frames dela (voto)."""
    return previsoes.groupby("tomada").agg(pessoa=("pessoa", "first"), real=("real", "first"),
                                           previsto=("previsto", lambda s: s.mode().iloc[0]))


def resumir_por_pessoa(previsoes):
    """Acerto por frame, F1 macro e acerto por tomada de cada pessoa, mais a média."""
    tomadas = votar_por_tomada(previsoes)
    linhas = []
    for pessoa, grupo in previsoes.groupby("pessoa", sort=False):
        da_pessoa = tomadas[tomadas["pessoa"] == pessoa]
        linhas.append({
            "pessoa": pessoa,
            "frames": len(grupo),
            "tomadas": len(da_pessoa),
            "acerto_frame": (grupo["real"] == grupo["previsto"]).mean(),
            "f1_macro": f1_score(grupo["real"], grupo["previsto"], average="macro"),
            "acerto_tomada": (da_pessoa["real"] == da_pessoa["previsto"]).mean(),
        })
    resumo = pd.DataFrame(linhas).set_index("pessoa")
    # Média simples das pessoas: cada pessoa pesa igual, tenha gravado mais ou menos
    resumo.loc["Média"] = resumo.mean()
    resumo.loc["Média", ["frames", "tomadas"]] = resumo[["frames", "tomadas"]].iloc[:-1].sum()
    return resumo


def acerto_split_aleatorio(todos):
    """O número ENGANOSO: frames sorteados, mesmas pessoas e tomadas no treino e no teste."""
    X, y = preparar(todos)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=SEMENTE, stratify=y)
    return (treinar(X_tr, y_tr).predict(X_te) == y_te).mean()


# =====================================================================
# 4. RESULTADOS (tabelas, figuras e relatório)
# =====================================================================
def matriz_confusao(previsoes):
    """% dos frames de cada letra real (linha) previstos como cada letra (coluna)."""
    tabela = pd.crosstab(previsoes["real"], previsoes["previsto"], normalize="index")
    classes = sorted(set(tabela.index) | set(tabela.columns))
    return tabela.reindex(index=classes, columns=classes, fill_value=0.0)


def desenhar_matriz_confusao(matriz, caminho):
    n = len(matriz)
    fig, ax = plt.subplots(figsize=(0.32 * n + 2.2, 0.32 * n + 1.2), dpi=200)
    mapa = LinearSegmentedColormap.from_list("azul", AZUIS)
    imagem = ax.imshow(matriz.to_numpy() * 100, cmap=mapa, vmin=0, vmax=100)

    # Separação fina entre as células, na cor do fundo
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1)
    ax.tick_params(which="minor", length=0)

    # Número só nas células relevantes (>= 10%), para não poluir
    for i in range(n):
        for j in range(n):
            valor = matriz.iat[i, j] * 100
            if valor >= 10:
                ax.text(j, i, f"{valor:.0f}", ha="center", va="center", fontsize=5.5,
                        color="white" if valor >= 45 else TINTA)

    ax.set_xticks(range(n), [c.upper() for c in matriz.columns], fontsize=7, color=TINTA_SECUNDARIA)
    ax.set_yticks(range(n), [c.upper() for c in matriz.index], fontsize=7, color=TINTA_SECUNDARIA)
    ax.tick_params(which="major", length=0)
    ax.set_xlabel("Letra prevista", color=TINTA_SECUNDARIA, fontsize=9)
    ax.set_ylabel("Letra real", color=TINTA_SECUNDARIA, fontsize=9)
    for borda in ax.spines.values():
        borda.set_visible(False)

    barra = fig.colorbar(imagem, ax=ax, fraction=0.04, pad=0.02)
    barra.set_label("% dos frames da letra real", color=TINTA_SECUNDARIA, fontsize=8)
    barra.ax.tick_params(labelsize=7, colors=TINTA_SECUNDARIA, length=0)
    barra.outline.set_visible(False)

    fig.tight_layout()
    fig.savefig(caminho, facecolor="white")
    plt.close(fig)


def desenhar_acerto_por_classe(por_classe, media, caminho):
    """Barras horizontais, da melhor (em cima) para a pior, com a média como referência."""
    ordenado = por_classe.sort_values()
    fig, ax = plt.subplots(figsize=(6.0, 0.22 * len(ordenado) + 1.0), dpi=200)
    posicoes = np.arange(len(ordenado))
    ax.barh(posicoes, ordenado.to_numpy() * 100, height=0.62, color=COR_BARRA)

    for y, valor in zip(posicoes, ordenado.to_numpy()):
        ax.text(valor * 100 + 1, y, pct(valor, 0), va="center", fontsize=6.5, color=TINTA_SECUNDARIA,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6}, zorder=3)

    ax.axvline(media * 100, color=TINTA_FRACA, linewidth=1, linestyle=(0, (3, 2)))
    ax.text(media * 100 + 1, len(ordenado) - 0.3, f"média {pct(media)}", fontsize=7,
            color=TINTA_FRACA, va="bottom")

    ax.set_yticks(posicoes, [c.upper() for c in ordenado.index], fontsize=7, color=TINTA_SECUNDARIA)
    ax.set_xlim(0, 108)
    ax.set_xticks(range(0, 101, 20), [f"{v}%" for v in range(0, 101, 20)], fontsize=7, color=TINTA_FRACA)
    ax.set_xlabel("Acerto por frame (pessoa de fora)", color=TINTA_SECUNDARIA, fontsize=8)
    ax.grid(axis="x", color=COR_GRADE, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for borda in ax.spines.values():
        borda.set_visible(False)

    fig.tight_layout()
    fig.savefig(caminho, facecolor="white")
    plt.close(fig)


def salvar_resultados(pasta, info_datasets, resumo, previsoes, fora_da_conta, acerto_aleatorio, caminho_modelo,
                      ms_por_quadro):
    """Grava em 'pasta' as tabelas CSV, as figuras PNG e o relatorio.md. Retorna os caminhos."""
    os.makedirs(pasta, exist_ok=True)
    caminhos = {nome: os.path.join(pasta, nome) for nome in (
        "relatorio.md", "acerto_por_pessoa.csv", "acerto_por_classe.csv",
        "matriz_confusao.csv", "matriz_confusao.png", "acerto_por_classe.png")}

    por_classe = previsoes.assign(certo=previsoes["real"] == previsoes["previsto"]).groupby("real")["certo"].mean()
    matriz = matriz_confusao(previsoes)
    erros = previsoes[previsoes["real"] != previsoes["previsto"]]
    confusoes = erros.groupby(["real", "previsto"]).size().sort_values(ascending=False).head(10)

    resumo.round(4).to_csv(caminhos["acerto_por_pessoa.csv"])
    por_classe.rename("acerto_frame").round(4).sort_values().to_csv(caminhos["acerto_por_classe.csv"])
    matriz.round(4).to_csv(caminhos["matriz_confusao.csv"])
    desenhar_matriz_confusao(matriz, caminhos["matriz_confusao.png"])
    desenhar_acerto_por_classe(por_classe, resumo.loc["Média", "acerto_frame"], caminhos["acerto_por_classe.png"])

    linhas = [
        "# Resultados do classificador de letras",
        "",
        f"Gerado por `treinar_classificador.py` em {datetime.datetime.now():%d/%m/%Y %H:%M}. "
        f"Modelo: {descricao_modelo()}; scikit-learn {sklearn.__version__}; classes: {CLASSES} ({len(por_classe)}).",
        "",
        "## Dados",
        "",
        "| Pessoa | Tomadas | Frames com mão | Frames sem mão (descartados) |",
        "|---|---:|---:|---:|",
        *[f"| {nome} | {i['tomadas']} | {i['frames']} | {i['frames_sem_mao']} |" for nome, i in info_datasets.items()],
        "",
        "## Avaliação \"pessoa de fora\"",
        "",
        "Para cada pessoa, o modelo é treinado com as outras e testado nela. "
        "O acerto por tomada é a letra mais prevista entre os frames da tomada.",
        "",
        "| Pessoa testada | Frames | Tomadas | Acerto por frame | F1 macro | Acerto por tomada |",
        "|---|---:|---:|---:|---:|---:|",
        *[f"| {'**Média**' if p == 'Média' else p} | {int(r.frames)} | {int(r.tomadas)} | {pct(r.acerto_frame)} | "
          f"{num(r.f1_macro)} | {pct(r.acerto_tomada)} |" for p, r in resumo.iterrows()],
        "",
    ]
    if any(fora_da_conta.values()):
        linhas += ["Classes que só a pessoa testada tem (fora da conta): " + "; ".join(
            f"{p}: {', '.join(c)}" for p, c in fora_da_conta.items() if c) + ".", ""]
    linhas += [
        "## Acerto por letra",
        "",
        "Da pior para a melhor (todas as pessoas testadas juntas): "
        + ", ".join(f"{c.upper()} {pct(v, 0)}" for c, v in por_classe.sort_values().items()) + ".",
        "",
        "Figuras: `acerto_por_classe.png` e `matriz_confusao.png`.",
        "",
        "## Confusões mais comuns",
        "",
        "| Letra real | Prevista como | Frames |",
        "|---|---|---:|",
        *[f"| {a.upper()} | {b.upper()} | {n} |" for (a, b), n in confusoes.items()],
        "",
        "## Comparação: frames sorteados (ilusório)",
        "",
        f"Sorteando 25% dos frames para teste, o acerto é {pct(acerto_aleatorio)}. O número é ilusório: "
        "frames da mesma tomada caem no treino e no teste. Não usar como resultado.",
        "",
        "## Modelo final",
        "",
        f"Treinado com todos os {int(resumo.loc['Média', 'frames'])} frames das {len(info_datasets)} pessoas "
        f"e salvo em `{os.path.relpath(caminho_modelo, SCRIPT_DIR).replace(os.sep, '/')}`. "
        f"Tempo para classificar um quadro: {num(ms_por_quadro, 2)} ms.",
        "",
    ]
    with open(caminhos["relatorio.md"], "w", encoding="utf-8") as arquivo:
        arquivo.write("\n".join(linhas))
    return caminhos


# =====================================================================
# PROGRAMA PRINCIPAL
# =====================================================================
def main():
    datasets, info_datasets = {}, {}
    for nome, pasta in DATASETS.items():
        dados, info = ler_dataset(nome, pasta)
        datasets[nome] = filtrar_classes(dados)
        info_datasets[nome] = info
        print(f"[DADOS] {nome}: {info['tomadas']} tomadas, {info['frames']} frames com mão "
              f"({info['frames_sem_mao']} sem mão descartados)")

    previsoes, fora_da_conta = avaliar_pessoa_de_fora(datasets)
    resumo = resumir_por_pessoa(previsoes)

    print(f"\nAvaliação 'pessoa de fora' ({CLASSES}):")
    print(f"  {'pessoa':10s} {'por frame':>10s} {'por tomada':>11s}")
    for pessoa, r in resumo.iterrows():
        print(f"  {pessoa:10s} {pct(r.acerto_frame):>10s} {pct(r.acerto_tomada):>11s}")

    todos = pd.concat(datasets.values(), ignore_index=True)
    acerto_aleatorio = acerto_split_aleatorio(todos)

    # Modelo final: todas as pessoas no treino (mais gente = melhor para pessoas novas)
    modelo = treinar(*preparar(todos))
    os.makedirs(PASTA_MODELOS, exist_ok=True)
    caminho_modelo = os.path.join(PASTA_MODELOS, f"classificador_{CLASSES}.joblib")
    ms_por_quadro = latencia_por_quadro_ms(modelo, preparar(todos)[0])
    joblib.dump({
        "modelo": modelo,
        "classes": [str(c) for c in modelo.classes_],
        "caracteristicas": caracteristicas_mao.NOMES,
        "proporcao_imagem": caracteristicas_mao.PROPORCAO_IMAGEM,
        "datasets": {nome: len(d) for nome, d in datasets.items()},
        "avaliacao_pessoa_de_fora": resumo.to_dict(orient="index"),
        "versao_sklearn": sklearn.__version__,
        "tipo_modelo": MODELO,
        "descricao": descricao_modelo(),
    }, caminho_modelo)

    caminhos = salvar_resultados(PASTA_RESULTADOS, info_datasets, resumo, previsoes, fora_da_conta,
                                 acerto_aleatorio, caminho_modelo, ms_por_quadro)
    print(f"\n[MODELO] {descricao_modelo()} | {num(ms_por_quadro, 2)} ms por quadro")
    print(f"[MODELO] {len(todos)} frames de {len(datasets)} pessoas -> {os.path.relpath(caminho_modelo, SCRIPT_DIR)}")
    print(f"[RESULTADOS] relatório, tabelas e figuras em {os.path.relpath(PASTA_RESULTADOS, SCRIPT_DIR)}"
          f"{os.sep} ({len(caminhos)} arquivos)")


if __name__ == "__main__":
    main()
