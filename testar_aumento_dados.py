"""
Testa se o aumento de dados (aumento_dados.py) melhora o acerto com pessoas novas.

Uso (de dentro da pasta do projeto):
  .venv\\Scripts\\python testar_aumento_dados.py

Avaliação "pessoa de fora" com o Random Forest de treinar_classificador.py. Os frames
artificiais são gerados SÓ a partir das pessoas de treino de cada rodada; a pessoa de
teste é avaliada só com frames reais. Saída em resultados/aumento_dados/.
"""

import os

import matplotlib
matplotlib.use("Agg")  # só gera arquivos, não abre janela
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import aumento_dados
import caracteristicas_mao
import treinar_classificador as treino

PASTA_SAIDA = os.path.join(treino.PASTA_RESULTADOS, "aumento_dados")
N_COPIAS = 3   # frames artificiais por frame real

# Nome -> intensidades (rotação em graus, proporção dos dedos, ruído); None = sem aumento
CONFIGURACOES = {
    "sem aumento": None,
    "só rotação": {"rotacao": aumento_dados.ROTACAO_MAX_GRAUS, "proporcao_dedos": 0, "ruido": 0},
    "só proporção dos dedos": {"rotacao": 0, "proporcao_dedos": aumento_dados.PROPORCAO_DEDOS, "ruido": 0},
    "só ruído": {"rotacao": 0, "proporcao_dedos": 0, "ruido": aumento_dados.RUIDO},
    "as três juntas": {"rotacao": aumento_dados.ROTACAO_MAX_GRAUS,
                       "proporcao_dedos": aumento_dados.PROPORCAO_DEDOS, "ruido": aumento_dados.RUIDO},
}


def ler_todos():
    partes = []
    for nome, pasta in treino.DATASETS.items():
        dados, _ = treino.ler_dataset(nome, pasta)
        partes.append(treino.filtrar_classes(dados).assign(pessoa=nome))
    todos = pd.concat(partes, ignore_index=True)
    pontos = todos[treino.COLUNAS_XYZ].to_numpy().reshape(-1, 21, 3)
    return (pontos, todos["esquerda"].to_numpy(), np.asarray(todos["label"], dtype="U"),
            todos["pessoa"].to_numpy(), todos["tomada"].to_numpy())


def avaliar(configuracao, pontos, esquerda, y, pessoa, tomada):
    """Previsões de todos os frames reais, cada pessoa prevista por um modelo que não a viu."""
    X_real = caracteristicas_mao.extrair(pontos, esquerda)
    partes = []
    for teste in dict.fromkeys(pessoa):
        tr, te = pessoa != teste, pessoa == teste
        X_tr, y_tr = X_real[tr], y[tr]
        if configuracao is not None:
            gerador = np.random.default_rng(treino.SEMENTE)
            artificiais, lado = aumento_dados.aumentar(pontos[tr], esquerda[tr], N_COPIAS, gerador, **configuracao)
            X_tr = np.vstack([X_tr, caracteristicas_mao.extrair(artificiais, lado)])
            y_tr = np.concatenate([y_tr, np.repeat(y[tr], N_COPIAS)])
        modelo = treino.treinar(X_tr, y_tr)
        partes.append(pd.DataFrame({"pessoa": teste, "tomada": tomada[te], "real": y[te],
                                    "previsto": modelo.predict(X_real[te])}))
    return pd.concat(partes, ignore_index=True)


def desenhar(medias, caminho):
    nomes = list(medias.index)[::-1]
    valores = medias.loc[nomes, "acerto_frame"].to_numpy() * 100
    base = medias.loc["sem aumento", "acerto_frame"] * 100
    fig, ax = plt.subplots(figsize=(6.2, 2.6), dpi=200)
    cores = [treino.TINTA_FRACA if n == "sem aumento" else treino.COR_BARRA for n in nomes]
    ax.barh(range(len(nomes)), valores, height=0.6, color=cores)
    for i, v in enumerate(valores):
        ax.text(v + 0.4, i, treino.num(v, 1) + "%", va="center", fontsize=7, color=treino.TINTA_SECUNDARIA)
    ax.axvline(base, color=treino.TINTA_FRACA, linewidth=1, linestyle=(0, (3, 2)))
    ax.set_yticks(range(len(nomes)), nomes, fontsize=8, color=treino.TINTA_SECUNDARIA)
    menor = max(0, int(min(valores) // 5 * 5) - 5)
    ax.set_xlim(menor, max(valores) + 6)
    ax.set_xlabel("Acerto por frame (pessoa de fora, média das 3 pessoas)", fontsize=8,
                  color=treino.TINTA_SECUNDARIA)
    ax.tick_params(length=0, labelsize=7, colors=treino.TINTA_FRACA)
    ax.grid(axis="x", color=treino.COR_GRADE, linewidth=0.6)
    ax.set_axisbelow(True)
    for borda in ax.spines.values():
        borda.set_visible(False)
    fig.tight_layout()
    fig.savefig(caminho, facecolor="white")
    plt.close(fig)


def main():
    pontos, esquerda, y, pessoa, tomada = ler_todos()
    resumos = {}
    for nome, configuracao in CONFIGURACOES.items():
        previsoes = avaliar(configuracao, pontos, esquerda, y, pessoa, tomada)
        resumos[nome] = treino.resumir_por_pessoa(previsoes)
        media = resumos[nome].loc["Média"]
        print(f"  {nome:24s} por frame {treino.pct(media.acerto_frame):>6s} | por tomada {treino.pct(media.acerto_tomada):>6s}")

    os.makedirs(PASTA_SAIDA, exist_ok=True)
    tabela = pd.concat({n: r for n, r in resumos.items()}, names=["configuracao", "pessoa"]).round(4)
    tabela.to_csv(os.path.join(PASTA_SAIDA, "acerto_por_configuracao.csv"))
    medias = tabela.xs("Média", level="pessoa")
    desenhar(medias, os.path.join(PASTA_SAIDA, "aumento_dados.png"))

    linhas = [
        "# Aumento de dados",
        "",
        f"Random Forest, avaliação \"pessoa de fora\". {N_COPIAS} frames artificiais por frame real, gerados só "
        f"a partir das pessoas de treino. Intensidades: rotação até {treino.num(aumento_dados.ROTACAO_MAX_GRAUS, 0)}°, "
        f"proporção dos dedos ±{treino.pct(aumento_dados.PROPORCAO_DEDOS, 0)}, ruído de "
        f"{treino.pct(aumento_dados.RUIDO, 0)} do tamanho da mão.",
        "",
        "| Configuração | Acerto por frame | F1 macro | Acerto por tomada |",
        "|---|---:|---:|---:|",
        *[f"| {n} | {treino.pct(r.acerto_frame)} | {treino.num(r.f1_macro)} | {treino.pct(r.acerto_tomada)} |"
          for n, r in medias.iterrows()],
        "",
        "## Por pessoa testada (acerto por frame)",
        "",
        "| Configuração | " + " | ".join(treino.DATASETS) + " |",
        "|---|" + "---:|" * len(treino.DATASETS),
        *[f"| {n} | " + " | ".join(treino.pct(resumos[n].loc[p, 'acerto_frame']) for p in treino.DATASETS) + " |"
          for n in CONFIGURACOES],
        "",
        "Figura: `aumento_dados.png`.",
        "",
    ]
    with open(os.path.join(PASTA_SAIDA, "relatorio.md"), "w", encoding="utf-8") as arquivo:
        arquivo.write("\n".join(linhas))
    print(f"\n[RESULTADOS] {os.path.relpath(PASTA_SAIDA, treino.SCRIPT_DIR)}{os.sep}")


if __name__ == "__main__":
    main()
