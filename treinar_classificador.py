"""
Treina o classificador (Random Forest) com os datasets do grupo.

Cada pasta em DATASETS é o dataset de UMA pessoa e continua separado:
  1. Avaliação "pessoa de fora": para cada pessoa, treina com as outras e
     testa nela. É o número que importa: acertar com alguém que o modelo
     nunca viu.
  2. Modelo final: depois da avaliação, treina com todas as pessoas e salva
     em modelos/ (mais gente no treino = modelo melhor para pessoas novas).

Por que não sortear frames para teste? Os 90 frames de uma tomada são quase
iguais entre si. Sorteando, o teste vira cópia do treino e o acerto fica
perto de 99%, mas é ilusório. O script mostra esse número só para comparação.
"""

import os
import re
import glob
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
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

# Tomadas com o label errado no CSV: (dataset, arquivo) -> label certo.
CORRECOES_LABEL = {
    # O CSV diz 'm', mas a tomada é quase igual à 'letra n_esquerdo.csv' e
    # diferente das tomadas de 'm' do Fernando: o label ficou da gravação anterior.
    ("Fernando", "letra n_direito.csv"): "n",
}

# Tomadas com a mão errada no nome do arquivo: (dataset, arquivo) -> mão certa.
# Critério: rodando o MediaPipe no .mp4 da tomada, 100% dos frames verificados
# foram vistos como a OUTRA mão (convenção em caracteristicas_mao.mao_esquerda).
# Quase todas têm um "_2" com a mão certa: a mão direita foi gravada com o TAB
# ainda em "esquerda" e depois regravada. 'letra r esquerda.csv' ficou de fora
# (8 de 15 frames concordam): não dá para afirmar que está errada.
CORRECOES_MAO = {
    ("Gabriel", "letra 1 esquerda.csv"): "direita",
    ("Gabriel", "letra 3 esquerda.csv"): "direita",
    ("Gabriel", "letra 6 esquerda.csv"): "direita",
    ("Gabriel", "letra 7 esquerda.csv"): "direita",
    ("Gabriel", "letra 9 esquerda.csv"): "direita",
    ("Gabriel", "letra e esquerda.csv"): "direita",
    ("Gabriel", "letra h esquerda.csv"): "direita",
    ("Gabriel", "letra i direita_2.csv"): "esquerda",
    ("Gabriel", "letra k esquerda.csv"): "direita",
    ("Gabriel", "letra k esquerda_2.csv"): "direita",
    ("Gabriel", "letra m esquerda.csv"): "direita",
    ("Gabriel", "letra p esquerda.csv"): "direita",
    ("Gabriel", "letra t esquerda.csv"): "direita",
    ("Gabriel", "letra w esquerda.csv"): "direita",
    ("Gabriel", "letra y esquerda.csv"): "direita",
}

N_ARVORES = 300
SEMENTE = 42
PASTA_MODELOS = os.path.join(SCRIPT_DIR, "modelos")

# Nome dos arquivos do coletor: "letra a direita.csv", "letra 1a direito.csv",
# "letra n_direito.csv", "letra j esquerda_2.csv"...
PADRAO_NOME = re.compile(r"^letra (\w)\w*?[ _](direit|esquerd)[ao](_\d+)?\.csv$")

COLUNAS_XYZ = [f"lm_{i}_{eixo}" for i in range(21) for eixo in ("x", "y", "z")]


def ler_dataset(nome, pasta):
    """Lê os CSVs de uma pessoa. Retorna um DataFrame só com os frames COM mão,
    com as colunas x/y/z dos landmarks, 'label', 'esquerda' e 'tomada'."""
    arquivos = sorted(glob.glob(os.path.join(pasta, "*.csv")))
    if not arquivos:
        raise SystemExit(f"[ERRO] Nenhum CSV em {pasta}")

    # Correções que citam um arquivo que não existe (renomeado ou apagado?)
    existentes = {os.path.basename(a) for a in arquivos}
    for dataset, arquivo in list(CORRECOES_LABEL) + list(CORRECOES_MAO):
        if dataset == nome and arquivo not in existentes:
            print(f"[AVISO] {nome}: a tabela de correção cita '{arquivo}', que não existe em {pasta}.")

    tomadas = []
    frames_sem_mao = 0
    maos_corrigidas = 0
    for caminho in arquivos:
        arquivo = os.path.basename(caminho)
        casamento = PADRAO_NOME.match(arquivo)
        if not casamento:
            print(f"[AVISO] {nome}: '{arquivo}' fora do padrão 'letra <x> <mão>.csv', ignorado.")
            continue
        letra_do_nome = casamento.group(1).lower()
        esquerda = casamento.group(2) == "esquerd"
        if (nome, arquivo) in CORRECOES_MAO:
            esquerda = CORRECOES_MAO[(nome, arquivo)] == "esquerda"
            maos_corrigidas += 1

        df = pd.read_csv(caminho, dtype={"label": str})
        faltando = [c for c in COLUNAS_XYZ + ["hand_confidence", "label"] if c not in df.columns]
        if faltando:
            print(f"[AVISO] {nome}: '{arquivo}' sem as colunas {faltando[:3]}..., ignorado.")
            continue

        labels = df["label"].str.lower().unique()
        label = CORRECOES_LABEL.get((nome, arquivo), labels[0])
        if len(labels) != 1 or (label != letra_do_nome and (nome, arquivo) not in CORRECOES_LABEL):
            print(f"[AVISO] {nome}: '{arquivo}' tem label {list(labels)} mas o nome diz '{letra_do_nome}', ignorado.")
            continue

        # Frames sem mão têm landmarks zerados e confiança 0: não ensinam nada
        com_mao = df["hand_confidence"] > 0
        frames_sem_mao += int((~com_mao).sum())
        df = df.loc[com_mao, COLUNAS_XYZ].assign(label=label, esquerda=esquerda, tomada=f"{nome}/{arquivo}")
        tomadas.append(df)

    dados = pd.concat(tomadas, ignore_index=True)
    print(f"[DATASET] {nome}: {len(tomadas)} tomadas, {len(dados)} frames com mão "
          f"({frames_sem_mao} sem mão descartados, {maos_corrigidas} tomadas com a mão corrigida)")
    return dados


def filtrar_classes(dados):
    letra = dados["label"].str.isalpha()
    if CLASSES == "letras":
        return dados[letra]
    if CLASSES == "numeros":
        return dados[~letra]
    return dados


def preparar(dados):
    """Converte um DataFrame de frames em (X, y) para o Random Forest."""
    pontos = dados[COLUNAS_XYZ].to_numpy().reshape(-1, 21, 3)
    X = caracteristicas_mao.extrair(pontos, dados["esquerda"].to_numpy())
    y = np.asarray(dados["label"], dtype="U")
    return X, y


def treinar(X, y):
    # Pesos por amostra compensam classes com mais tomadas que outras.
    # (O class_weight="balanced" do scikit-learn 1.9.1 quebra com labels como '1', '2';
    #  por isso os pesos vão no fit.)
    modelo = RandomForestClassifier(n_estimators=N_ARVORES, n_jobs=-1, random_state=SEMENTE)
    modelo.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    return modelo


def avaliar_pessoa_de_fora(datasets):
    """Para cada pessoa: treina com as outras, testa nela. Retorna o resumo por pessoa."""
    print("\n" + "=" * 70)
    print("AVALIAÇÃO 'PESSOA DE FORA' (treina com as outras, testa na pessoa)")
    print("=" * 70)

    resumo = {}
    acertos_por_classe = {}
    for teste, dados_teste in datasets.items():
        treino = pd.concat([d for nome, d in datasets.items() if nome != teste], ignore_index=True)
        modelo = treinar(*preparar(treino))

        # Classe que não existe no treino não tem como ser acertada: fica fora da conta
        sem_treino = sorted(set(dados_teste["label"]) - set(modelo.classes_))
        dados_teste = dados_teste[~dados_teste["label"].isin(sem_treino)]
        X, y = preparar(dados_teste)
        previsto = modelo.predict(X)

        # Acerto por tomada: a classe mais prevista nos frames da tomada (voto)
        votos = pd.DataFrame({"tomada": dados_teste["tomada"].to_numpy(), "real": y, "previsto": previsto})
        por_tomada = votos.groupby("tomada").agg(real=("real", "first"),
                                                 previsto=("previsto", lambda s: s.mode().iloc[0]))

        resumo[teste] = {
            "frames": len(y),
            "acerto_frame": accuracy_score(y, previsto),
            "f1_macro": f1_score(y, previsto, average="macro"),
            "acerto_tomada": (por_tomada["real"] == por_tomada["previsto"]).mean(),
            "tomadas": len(por_tomada),
        }
        r = resumo[teste]
        print(f"\n[TESTE: {teste}] treino: {', '.join(n for n in datasets if n != teste)}")
        print(f"  {r['frames']} frames | acerto por frame: {r['acerto_frame']:.1%} | F1 macro: {r['f1_macro']:.2f} "
              f"| acerto por tomada (voto): {r['acerto_tomada']:.1%} de {r['tomadas']}")
        if sem_treino:
            print(f"  classes que só esta pessoa tem (fora da conta): {', '.join(sem_treino)}")

        erros = votos[votos["real"] != votos["previsto"]]
        confusoes = erros.groupby(["real", "previsto"]).size().sort_values(ascending=False).head(8)
        print("  mais confundidas (real -> previsto, frames): "
              + ", ".join(f"{a}->{b} ({n})" for (a, b), n in confusoes.items()))

        for real, certo in zip(y, y == previsto):
            acertos_por_classe.setdefault(real, []).append(certo)

    media = {k: np.mean([r[k] for r in resumo.values()]) for k in ("acerto_frame", "f1_macro", "acerto_tomada")}
    print(f"\n[MÉDIA DAS PESSOAS] acerto por frame: {media['acerto_frame']:.1%} | F1 macro: {media['f1_macro']:.2f} "
          f"| acerto por tomada: {media['acerto_tomada']:.1%}")
    print("[ACERTO POR CLASSE] (todas as pessoas, da pior para a melhor)")
    ordenadas = sorted(acertos_por_classe.items(), key=lambda kv: np.mean(kv[1]))
    print("  " + "  ".join(f"{classe}:{np.mean(certos):.0%}" for classe, certos in ordenadas))
    return resumo


def comparar_split_aleatorio(todos):
    """O número ENGANOSO: frames sorteados, mesmas pessoas e tomadas no treino e no teste."""
    X, y = preparar(todos)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=SEMENTE, stratify=y)
    acerto = accuracy_score(y_te, treinar(X_tr, y_tr).predict(X_te))
    print(f"\n[SÓ PARA COMPARAR] frames sorteados (25% teste): {acerto:.1%} "
          "- ilusório, não usar como resultado")


def main():
    datasets = {nome: filtrar_classes(ler_dataset(nome, pasta)) for nome, pasta in DATASETS.items()}

    todas = sorted(set().union(*(set(d["label"]) for d in datasets.values())))
    print(f"\n[CLASSES] '{CLASSES}': {len(todas)} classes -> {' '.join(todas)}")
    for nome, dados in datasets.items():
        faltam = sorted(set(todas) - set(dados["label"]))
        if faltam:
            print(f"  {nome} não tem: {' '.join(faltam)}")

    resumo = avaliar_pessoa_de_fora(datasets)

    todos = pd.concat(datasets.values(), ignore_index=True)
    comparar_split_aleatorio(todos)

    # Modelo final: todas as pessoas no treino
    modelo = treinar(*preparar(todos))
    os.makedirs(PASTA_MODELOS, exist_ok=True)
    caminho = os.path.join(PASTA_MODELOS, f"random_forest_{CLASSES}.joblib")
    joblib.dump({
        "modelo": modelo,
        "classes": [str(c) for c in modelo.classes_],
        "caracteristicas": caracteristicas_mao.NOMES,
        "proporcao_imagem": caracteristicas_mao.PROPORCAO_IMAGEM,
        "datasets": {nome: len(d) for nome, d in datasets.items()},
        "avaliacao_pessoa_de_fora": resumo,
        "versao_sklearn": sklearn.__version__,
    }, caminho)
    print(f"\n[MODELO FINAL] treinado com {len(todos)} frames de {len(datasets)} pessoas -> {caminho}")


if __name__ == "__main__":
    main()
