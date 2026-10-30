"""
Compara Random Forest, SVM e MLP com a avaliação "pessoa de fora".

Uso (de dentro da pasta do projeto):
  .venv\\Scripts\\python comparar_modelos.py

Usa os mesmos dados e as mesmas 35 características de treinar_classificador.py.
Para cada pessoa de teste, os hiperparâmetros de cada modelo são escolhidos só com as
outras pessoas (treina com uma, valida na outra); o modelo escolhido é então treinado com
as duas e testado na pessoa de fora. Assim a pessoa de teste não influencia nenhuma escolha.

Saída em resultados/comparacao/: relatório, tabelas (CSV) e figura (PNG).
"""

import itertools
import os
import time

import matplotlib
matplotlib.use("Agg")  # só gera arquivos, não abre janela
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

import treinar_classificador as treino

PASTA_SAIDA = os.path.join(treino.PASTA_RESULTADOS, "comparacao")
SEMENTE = treino.SEMENTE

COMITE = "Comitê (média dos 3)"

# Cores das séries: slots da paleta categórica, na ordem fixa
CORES = {"Random Forest": "#2a78d6", "SVM": "#eb6834", "MLP": "#1baf7a", COMITE: "#eda100"}


def grade(**opcoes):
    """{"C": [1, 10], "gamma": [0.01]} -> [{"C": 1, "gamma": 0.01}, {"C": 10, "gamma": 0.01}]"""
    nomes = list(opcoes)
    return [dict(zip(nomes, valores)) for valores in itertools.product(*opcoes.values())]


# Modelo -> (função que cria o modelo com os hiperparâmetros, grade de hiperparâmetros)
MODELOS = {
    "Random Forest": (
        lambda **p: RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEMENTE, **p),
        grade(max_features=["sqrt", 0.5], min_samples_leaf=[1, 5]),
    ),
    "SVM": (
        # RBF precisa de características na mesma escala
        lambda **p: make_pipeline(StandardScaler(), SVC(kernel="rbf", **p)),
        grade(C=[0.1, 0.3, 1, 3, 10], gamma=[0.001, 0.003, 0.01, 0.03]),
    ),
    "MLP": (
        lambda **p: make_pipeline(StandardScaler(), MLPClassifier(
            early_stopping=True, max_iter=500, random_state=SEMENTE, **p)),
        grade(hidden_layer_sizes=[(128, 64), (256, 128), (512, 256)], alpha=[1e-4, 1e-2, 1e-1]),
    ),
}


# Opções que só valem no treino final de cada rodada (não na escolha de hiperparâmetros):
# o comitê precisa das probabilidades do SVM.
OPCOES_FINAIS = {"SVM": {"probability": True, "random_state": SEMENTE}}

NOMES = list(MODELOS) + [COMITE]


def ajustar(modelo, X, y):
    """Treina com pesos por amostra (compensam letras com mais tomadas que outras)."""
    pesos = compute_sample_weight("balanced", y)
    if isinstance(modelo, Pipeline):
        modelo.fit(X, y, **{f"{modelo.steps[-1][0]}__sample_weight": pesos})
    else:
        modelo.fit(X, y, sample_weight=pesos)
    return modelo


def preparar_todos():
    """Lê os datasets e calcula as características de todos os frames uma vez só."""
    partes = []
    for nome, pasta in treino.DATASETS.items():
        dados, info = treino.ler_dataset(nome, pasta)
        partes.append(treino.filtrar_classes(dados).assign(pessoa=nome))
        print(f"[DADOS] {nome}: {info['tomadas']} tomadas, {info['frames']} frames com mão")
    todos = pd.concat(partes, ignore_index=True)
    X, y = treino.preparar(todos)
    return X, y, todos["pessoa"].to_numpy(), todos["tomada"].to_numpy()


def escolher_hiperparametros(nome, X, y, pessoa, pessoas_treino):
    """Validação interna: treina com uma pessoa de treino e valida na outra (e vice-versa)."""
    criar, opcoes = MODELOS[nome]
    melhor, melhor_acerto = None, -1.0
    for parametros in opcoes:
        acertos = []
        for validacao in pessoas_treino:
            tr = np.isin(pessoa, pessoas_treino) & (pessoa != validacao)
            va = pessoa == validacao
            modelo = ajustar(criar(**parametros), X[tr], y[tr])
            acertos.append((modelo.predict(X[va]) == y[va]).mean())
        if np.mean(acertos) > melhor_acerto:
            melhor, melhor_acerto = parametros, float(np.mean(acertos))
    return melhor, melhor_acerto


def latencia_por_frame_ms(modelo, X, repeticoes=200):
    """Tempo para classificar UM frame (como no tempo real), em milissegundos."""
    if hasattr(modelo, "n_jobs"):
        modelo.n_jobs = 1   # um frame por vez: paralelizar custa mais do que economiza
    linha = X[:1]
    modelo.predict(linha)
    inicio = time.perf_counter()
    for i in range(repeticoes):
        modelo.predict(X[i % len(X):i % len(X) + 1])
    return (time.perf_counter() - inicio) / repeticoes * 1000


def comparar(X, y, pessoa, tomada):
    pessoas = list(dict.fromkeys(pessoa))
    linhas, previsoes = [], []

    def registrar(nome, teste, previsto, **dados):
        te = pessoa == teste
        previsoes.append(pd.DataFrame({"modelo": nome, "pessoa": teste, "tomada": tomada[te],
                                       "real": y[te], "previsto": previsto}))
        linhas.append({"modelo": nome, "pessoa": teste, **dados})
        print(f"  {nome:20s} teste={teste:9s} acerto por frame {treino.pct((previsto == y[te]).mean())} "
              f"| {dados['hiperparametros']}")

    for teste in pessoas:
        treino_pessoas = [p for p in pessoas if p != teste]
        tr, te = pessoa != teste, pessoa == teste
        probabilidades, segundos, latencias = [], 0.0, 0.0
        for nome, (criar, _) in MODELOS.items():
            parametros, acerto_validacao = escolher_hiperparametros(nome, X, y, pessoa, treino_pessoas)
            inicio = time.perf_counter()
            modelo = ajustar(criar(**parametros, **OPCOES_FINAIS.get(nome, {})), X[tr], y[tr])
            segundos_treino = time.perf_counter() - inicio
            probabilidades.append(modelo.predict_proba(X[te]))
            latencia = latencia_por_frame_ms(modelo, X[te])
            segundos, latencias = segundos + segundos_treino, latencias + latencia
            registrar(nome, teste, modelo.predict(X[te]), hiperparametros=parametros,
                      acerto_validacao_interna=acerto_validacao, segundos_treino=segundos_treino,
                      ms_por_frame=latencia)

        # Comitê: média das probabilidades dos três (todos usam as classes na mesma ordem)
        previsto = modelo.classes_[np.mean(probabilidades, axis=0).argmax(axis=1)]
        registrar(COMITE, teste, previsto, hiperparametros="os escolhidos acima para cada modelo",
                  acerto_validacao_interna=np.nan, segundos_treino=segundos, ms_por_frame=latencias)
    return pd.DataFrame(linhas), pd.concat(previsoes, ignore_index=True)


def resumir(execucoes, previsoes):
    """Uma linha por (modelo, pessoa) com as métricas, mais a média de cada modelo."""
    partes = []
    for nome, grupo in previsoes.groupby("modelo", sort=False):
        resumo = treino.resumir_por_pessoa(grupo.drop(columns="modelo"))
        partes.append(resumo.reset_index().assign(modelo=nome))
    tabela = pd.concat(partes, ignore_index=True)
    tempos = execucoes.groupby("modelo")[["segundos_treino", "ms_por_frame"]].mean()
    return tabela, tempos


def desenhar(tabela, caminho):
    """Dois painéis (por frame, por tomada): barras por pessoa, uma cor por modelo."""
    pessoas = list(dict.fromkeys(tabela["pessoa"]))
    modelos = NOMES
    largura = 0.2
    fig, eixos = plt.subplots(1, 2, figsize=(9.5, 3.8), dpi=200, sharey=True)
    for ax, (coluna, titulo) in zip(eixos, (("acerto_frame", "Acerto por frame"),
                                            ("acerto_tomada", "Acerto por tomada"))):
        for i, modelo in enumerate(modelos):
            valores = [tabela.loc[(tabela["modelo"] == modelo) & (tabela["pessoa"] == p), coluna].iloc[0] * 100
                       for p in pessoas]
            posicoes = np.arange(len(pessoas)) + (i - (len(modelos) - 1) / 2) * largura
            ax.bar(posicoes, valores, width=largura - 0.03, color=CORES[modelo], label=modelo)
            for x, v in zip(posicoes, valores):
                ax.text(x, v + 1.2, f"{v:.0f}", ha="center", fontsize=6, color=treino.TINTA_SECUNDARIA)
        ax.set_title(titulo, fontsize=9, color=treino.TINTA, loc="left")
        ax.set_xticks(range(len(pessoas)), pessoas, fontsize=8, color=treino.TINTA_SECUNDARIA)
        ax.set_ylim(0, 100)
        ax.set_yticks(range(0, 101, 20), [f"{v}%" for v in range(0, 101, 20)], fontsize=7,
                      color=treino.TINTA_FRACA)
        ax.grid(axis="y", color=treino.COR_GRADE, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        for borda in ax.spines.values():
            borda.set_visible(False)
    eixos[0].legend(frameon=False, fontsize=8, loc="upper left", ncols=4, bbox_to_anchor=(0, 1.16))
    fig.tight_layout()
    fig.savefig(caminho, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def salvar(execucoes, previsoes, tabela, tempos):
    os.makedirs(PASTA_SAIDA, exist_ok=True)
    tabela.round(4).to_csv(os.path.join(PASTA_SAIDA, "acerto_por_modelo.csv"), index=False)
    execucoes.round(4).to_csv(os.path.join(PASTA_SAIDA, "hiperparametros_e_tempos.csv"), index=False)
    por_letra = (previsoes.assign(certo=previsoes["real"] == previsoes["previsto"])
                 .pivot_table(index="real", columns="modelo", values="certo", aggfunc="mean")[NOMES])
    por_letra = por_letra.sort_values(NOMES[0])
    por_letra.round(4).to_csv(os.path.join(PASTA_SAIDA, "acerto_por_letra.csv"))
    desenhar(tabela, os.path.join(PASTA_SAIDA, "comparacao_modelos.png"))

    medias = tabela[tabela["pessoa"] == "Média"].set_index("modelo")
    linhas = [
        "# Comparação: Random Forest, SVM e MLP",
        "",
        "Avaliação \"pessoa de fora\" (3 rodadas). Os hiperparâmetros de cada modelo foram escolhidos "
        "só com as pessoas de treino de cada rodada. O comitê faz a média das probabilidades dos três modelos "
        "de cada rodada (o tempo dele é a soma dos três).",
        "",
        "| Modelo | Acerto por frame | F1 macro | Acerto por tomada | Treino (s) | Classificar 1 frame (ms) |",
        "|---|---:|---:|---:|---:|---:|",
        *[f"| {m} | {treino.pct(medias.loc[m, 'acerto_frame'])} | {treino.num(medias.loc[m, 'f1_macro'])} | "
          f"{treino.pct(medias.loc[m, 'acerto_tomada'])} | {treino.num(tempos.loc[m, 'segundos_treino'], 1)} | "
          f"{treino.num(tempos.loc[m, 'ms_por_frame'], 2)} |" for m in NOMES],
        "",
        "## Por pessoa testada",
        "",
        "| Modelo | Pessoa | Acerto por frame | F1 macro | Acerto por tomada |",
        "|---|---|---:|---:|---:|",
        *[f"| {r.modelo} | {r.pessoa} | {treino.pct(r.acerto_frame)} | {treino.num(r.f1_macro)} | "
          f"{treino.pct(r.acerto_tomada)} |" for r in tabela.itertuples() if r.pessoa != "Média"],
        "",
        "## Hiperparâmetros escolhidos em cada rodada",
        "",
        *[f"- {r.modelo}, teste {r.pessoa}: `{r.hiperparametros}` "
          f"(acerto na validação interna {treino.pct(r.acerto_validacao_interna)})" for r in execucoes.itertuples()],
        "",
        "## Acerto por letra (todas as pessoas testadas juntas)",
        "",
        "| Letra | " + " | ".join(NOMES) + " |",
        "|---|" + "---:|" * len(NOMES),
        *[f"| {letra.upper()} | " + " | ".join(treino.pct(v, 0) for v in linha) + " |"
          for letra, linha in por_letra.iterrows()],
        "",
        "Figura: `comparacao_modelos.png`.",
        "",
    ]
    with open(os.path.join(PASTA_SAIDA, "relatorio.md"), "w", encoding="utf-8") as arquivo:
        arquivo.write("\n".join(linhas))


def main():
    X, y, pessoa, tomada = preparar_todos()
    print("\nComparando (escolha de hiperparâmetros + teste na pessoa de fora):")
    execucoes, previsoes = comparar(X, y, pessoa, tomada)
    tabela, tempos = resumir(execucoes, previsoes)

    print(f"\n  {'modelo':20s} {'por frame':>10s} {'por tomada':>11s} {'1 frame':>9s}")
    for modelo in NOMES:
        r = tabela[(tabela["modelo"] == modelo) & (tabela["pessoa"] == "Média")].iloc[0]
        print(f"  {modelo:20s} {treino.pct(r.acerto_frame):>10s} {treino.pct(r.acerto_tomada):>11s} "
              f"{treino.num(tempos.loc[modelo, 'ms_por_frame'], 2):>6s} ms")

    salvar(execucoes, previsoes, tabela, tempos)
    print(f"\n[RESULTADOS] {os.path.relpath(PASTA_SAIDA, treino.SCRIPT_DIR)}{os.sep}")


if __name__ == "__main__":
    main()
