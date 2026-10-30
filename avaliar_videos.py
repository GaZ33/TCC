"""
Avalia vídeos gravados por voluntários com o mesmo processamento do sistema em tempo real.

Uso (de dentro da pasta do projeto):
  .venv\\Scripts\\python avaliar_videos.py <pasta_com_os_videos>

Nome de cada vídeo: <voluntario>_<etapa>_<letra>.<extensão>, por exemplo
  ana_imagem_L.mp4  ou  ana_video_Z.mov      (etapa: "imagem" ou "video")

Para cada vídeo:
  1. lê os quadros (reduzindo para ~30 por segundo) e detecta a mão com o MediaPipe, como o emissor;
  2. descobre se o vídeo está espelhado: o celular costuma salvar o vídeo SEM espelhar, e o sistema
     trabalha com a imagem espelhada (como a câmera do computador mostra). Como os voluntários usam a
     mão direita, a mão detectada pelo MediaPipe mostra qual é o caso;
  3. passa os quadros pelo classificador e pelo filtro do sistema (classificador.py e filtro.py), com o
     modelo de modelos/classificador_letras.joblib, simulando o sistema já ligado: a cada STEP quadros
     novos, uma janela vai ao classificador e a previsão vai ao filtro;
  4. registra a primeira letra mostrada (modo MANUAL do exibidor), o tempo até ela e a letra mais mostrada.

Saída: resultados/voluntarios/por_video.csv e resultados/voluntarios/relatorio.md.
Os vídeos não são copiados nem alterados.
"""

import argparse
import collections
import contextlib
import io
import logging
import os
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

import caracteristicas_mao
import classificador
import filtro

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PASTA_SAIDA = os.path.join(SCRIPT_DIR, "resultados", "voluntarios")
MODEL_PATH = os.path.join(SCRIPT_DIR, "hand_landmarker.task")
URL_MODELO_MAO = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/"
                  "float16/1/hand_landmarker.task")

MAO_DOS_VOLUNTARIOS = "direita"
QUADROS_POR_SEGUNDO = 30   # o emissor trabalha a ~30 quadros por segundo
TAMANHO_JANELA = 120       # mesma janela do emissor
STEP = 30                  # regime nominal do emissor (vídeos de uma letra quase não mudam de regime)
MAIOR_LADO = 960           # vídeos maiores são reduzidos (mais rápido; os pontos são normalizados)
EXTENSOES = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp")
ETAPAS = ("imagem", "video")
COLUNAS = classificador.COLUNA_FRAME, classificador.COLUNA_CONFIANCA


# =====================================================================
# 1. QUADROS E PONTOS DA MÃO
# =====================================================================
def criar_detector():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(URL_MODELO_MAO, MODEL_PATH)
    opcoes = vision.HandLandmarkerOptions(base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
                                          num_hands=1, running_mode=vision.RunningMode.IMAGE)
    return vision.HandLandmarker.create_from_options(opcoes)


def extrair_pontos(caminho, detector):
    """Lê o vídeo, espelha cada quadro (como o emissor faz com a câmera) e detecta a mão.

    Retorna (pontos (n, 21, 3), confianca (n,), nomes (n,) "Left"/"Right"/None, proporcao).
    Quadros sem mão têm pontos zerados e confiança 0, como no emissor.
    """
    captura = cv2.VideoCapture(caminho)
    if not captura.isOpened():
        raise ValueError(f"não foi possível abrir {caminho}")
    fps = captura.get(cv2.CAP_PROP_FPS) or QUADROS_POR_SEGUNDO
    passo = max(1, round(fps / QUADROS_POR_SEGUNDO))
    pontos, confianca, nomes, proporcao, i = [], [], [], None, 0
    while True:
        ok, quadro = captura.read()
        if not ok:
            break
        i += 1
        if (i - 1) % passo:
            continue
        altura, largura = quadro.shape[:2]
        if max(altura, largura) > MAIOR_LADO:
            escala = MAIOR_LADO / max(altura, largura)
            quadro = cv2.resize(quadro, (int(largura * escala), int(altura * escala)))
            altura, largura = quadro.shape[:2]
        proporcao = largura / altura
        quadro = cv2.flip(quadro, 1)
        imagem = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(quadro, cv2.COLOR_BGR2RGB))
        resultado = detector.detect(imagem)
        if resultado.hand_landmarks:
            pontos.append([(lm.x, lm.y, lm.z) for lm in resultado.hand_landmarks[0]])
            confianca.append(resultado.handedness[0][0].score)
            nomes.append(resultado.handedness[0][0].category_name)
        else:
            pontos.append(np.zeros((21, 3)))
            confianca.append(0.0)
            nomes.append(None)
    captura.release()
    if not pontos:
        raise ValueError(f"{caminho} não tem quadros")
    return np.array(pontos, dtype=float), np.array(confianca), nomes, proporcao


def corrigir_espelhamento(pontos, nomes, mao_esperada=MAO_DOS_VOLUNTARIOS):
    """Se a maioria dos quadros mostra a outra mão, o vídeo já estava espelhado: desfaz.

    Retorna (pontos, nomes, espelhado). Espelhar os pontos (x -> 1 - x) equivale a não ter
    espelhado o quadro, e o nome da mão troca junto.
    """
    vistos = [n for n in nomes if n is not None]
    if not vistos:
        return pontos, nomes, False
    esquerda_esperada = mao_esperada == "esquerda"
    contrarios = sum(caracteristicas_mao.mao_esquerda(n) != esquerda_esperada for n in vistos)
    if contrarios <= len(vistos) / 2:
        return pontos, nomes, False
    pontos = pontos.copy()
    com_mao = np.array([n is not None for n in nomes])
    pontos[com_mao, :, 0] = 1.0 - pontos[com_mao, :, 0]
    trocado = {"Left": "Right", "Right": "Left", None: None}
    return pontos, [trocado[n] for n in nomes], True


# =====================================================================
# 2. CADEIA DO SISTEMA (classificador + filtro)
# =====================================================================
def montar_janela(sequencia, ids, pontos, confianca, nomes, proporcao):
    """Mesmo pacote que o emissor envia, já decodificado pelo classificador."""
    linhas = [[fid, *np.asarray(p).reshape(-1).tolist(), c] for fid, p, c in zip(ids, pontos, confianca)]
    colunas = ([COLUNAS[0]] + [f"lm_{i}_{e}" for i in range(21) for e in ("x", "y", "z")] + [COLUNAS[1]])
    pacote = {"sequencia": sequencia, "colunas": colunas, "amostras": linhas,
              "maos": list(nomes), "proporcao": proporcao}
    return classificador.decodificar_pacote(pacote, mao_padrao_esquerda=False, proporcao_padrao=proporcao)


def simular_cadeia(modelo, pontos, confianca, nomes, proporcao, step=STEP):
    """Simula o sistema já ligado e sem mão na tela quando o vídeo começa.

    Retorna a lista de (indice_do_ultimo_quadro, letra) de cada letra mostrada pelo filtro.
    """
    pipeline = classificador.PipelineClassificacao(modelo, "janela")
    filtro_sistema = filtro.Filtro()
    n = len(pontos)
    vazio = np.zeros((TAMANHO_JANELA, 21, 3))
    # Antes do vídeo: uma janela só de quadros sem mão (o sistema estava ligado, sem ninguém)
    pipeline.processar(montar_janela(1, np.arange(1, TAMANHO_JANELA + 1), vazio, np.zeros(TAMANHO_JANELA),
                                     [None] * TAMANHO_JANELA, proporcao))
    todos_ids = np.arange(TAMANHO_JANELA + 1, TAMANHO_JANELA + 1 + n)
    mostradas, sequencia = [], 1
    for fim in range(step, n + 1, step):
        inicio = max(0, fim - TAMANHO_JANELA)
        sequencia += 1
        janela = montar_janela(sequencia, todos_ids[inicio:fim], pontos[inicio:fim], confianca[inicio:fim],
                               nomes[inicio:fim], proporcao)
        for previsao in pipeline.processar(janela):
            with contextlib.redirect_stderr(io.StringIO()):
                evento = filtro_sistema.processar_previsao(filtro.extrair_probabilidades(previsao))
            if evento is not None and evento["acao"] == "SHOW":
                mostradas.append((fim - 1, evento["valor"]))
    return mostradas


def avaliar_video(caminho, detector, modelo):
    nome = os.path.splitext(os.path.basename(caminho))[0]
    voluntario, etapa, letra = nome.rsplit("_", 2)
    pontos, confianca, nomes, proporcao = extrair_pontos(caminho, detector)
    pontos, nomes, espelhado = corrigir_espelhamento(pontos, nomes)
    mostradas = simular_cadeia(modelo, pontos, confianca, nomes, proporcao)

    # Acerto quadro a quadro (sem janela nem filtro), para comparar com o treino
    com_mao = confianca > 0
    acerto_quadros, mais_prevista = np.nan, ""
    if com_mao.any():
        esquerda = np.array([caracteristicas_mao.mao_esquerda(x) for x in np.array(nomes, dtype=object)[com_mao]])
        probabilidades, validos = modelo.inferir(pontos[com_mao], esquerda, proporcao)
        if len(probabilidades):
            previstas = np.array(modelo.classes)[probabilidades.argmax(axis=1)]
            acerto_quadros = float((previstas == letra.lower()).mean())
            mais_prevista = collections.Counter(previstas).most_common(1)[0][0]

    contagem = collections.Counter(l for _, l in mostradas)
    return {
        "arquivo": os.path.basename(caminho), "voluntario": voluntario, "etapa": etapa.lower(),
        "letra": letra.lower(), "espelhado": espelhado, "quadros": len(pontos),
        "fracao_com_mao": round(float(com_mao.mean()), 3),
        "primeira_letra": mostradas[0][1] if mostradas else "",
        "segundos_ate_primeira": round((mostradas[0][0] + 1) / QUADROS_POR_SEGUNDO, 2) if mostradas else np.nan,
        "letra_mais_mostrada": contagem.most_common(1)[0][0] if contagem else "",
        "vezes_mostrada": sum(contagem.values()),
        "acerto_quadros": round(acerto_quadros, 3) if not np.isnan(acerto_quadros) else np.nan,
        "letra_mais_prevista_quadros": mais_prevista,
    }


# =====================================================================
# 3. RELATÓRIO
# =====================================================================
def pct(valor):
    return f"{valor * 100:.0f}%" if not pd.isna(valor) else "–"


def salvar(tabela, pasta):
    os.makedirs(pasta, exist_ok=True)
    tabela.to_csv(os.path.join(pasta, "por_video.csv"), index=False)
    tabela = tabela.assign(acertou=tabela["primeira_letra"] == tabela["letra"])
    linhas = ["# Teste com voluntários", "",
              "Resposta do sistema = primeira letra mostrada (modo MANUAL do exibidor).", "",
              "## Por etapa", "",
              "| Etapa | Vídeos | Acerto | Nenhuma letra mostrada | Tempo médio até a letra |",
              "|---|---:|---:|---:|---:|"]
    for etapa, grupo in tabela.groupby("etapa"):
        tempo = grupo["segundos_ate_primeira"].mean()
        tempo = f"{tempo:.1f} s".replace(".", ",") if not pd.isna(tempo) else "–"
        linhas.append(f"| {etapa} | {len(grupo)} | {pct(grupo['acertou'].mean())} | "
                      f"{pct((grupo['primeira_letra'] == '').mean())} | {tempo} |")
    linhas += ["", "## Por letra e etapa (acerto da primeira letra mostrada)", "",
               "| Letra | " + " | ".join(ETAPAS) + " |", "|---|" + "---:|" * len(ETAPAS)]
    for letra, grupo in tabela.groupby("letra"):
        celulas = []
        for etapa in ETAPAS:
            g = grupo[grupo["etapa"] == etapa]
            celulas.append(f"{int(g['acertou'].sum())}/{len(g)}" if len(g) else "–")
        linhas.append(f"| {letra.upper()} | " + " | ".join(celulas) + " |")
    linhas += ["", "## Por vídeo", "",
               "| Arquivo | Primeira letra | Tempo | Mais mostrada | Acerto por quadro | Quadros com mão | Espelhado |",
               "|---|---|---:|---|---:|---:|---|"]
    for r in tabela.itertuples():
        tempo = f"{r.segundos_ate_primeira:.1f} s".replace(".", ",") if not pd.isna(r.segundos_ate_primeira) else "–"
        linhas.append(f"| {r.arquivo} | {r.primeira_letra.upper() or 'nenhuma'} | {tempo} | "
                      f"{r.letra_mais_mostrada.upper() or '–'} | {pct(r.acerto_quadros)} | {pct(r.fracao_com_mao)} | "
                      f"{'sim' if r.espelhado else 'não'} |")
    with open(os.path.join(pasta, "relatorio.md"), "w", encoding="utf-8") as arquivo:
        arquivo.write("\n".join(linhas) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Avalia vídeos de voluntários com o sistema.")
    parser.add_argument("pasta", help="pasta com os vídeos (<voluntario>_<etapa>_<letra>.mp4)")
    parser.add_argument("--saida", default=PASTA_SAIDA, help="pasta do relatório")
    args = parser.parse_args(argv)
    logging.disable(logging.WARNING)   # o filtro e o classificador avisam a cada janela

    videos = sorted(os.path.join(args.pasta, a) for a in os.listdir(args.pasta)
                    if a.lower().endswith(EXTENSOES) and a.count("_") >= 2)
    if not videos:
        raise SystemExit(f"[ERRO] Nenhum vídeo <voluntario>_<etapa>_<letra> em {args.pasta}")
    modelo = classificador.carregar_modelo(classificador.CAMINHO_MODELO_PADRAO)
    detector = criar_detector()

    linhas = []
    for caminho in videos:
        try:
            r = avaliar_video(caminho, detector, modelo)
        except ValueError as erro:
            print(f"[AVISO] {os.path.basename(caminho)}: {erro}")
            continue
        linhas.append(r)
        print(f"{r['arquivo']:32s} primeira: {r['primeira_letra'].upper() or 'nenhuma':8s} "
              f"mais mostrada: {r['letra_mais_mostrada'].upper() or '-':2s} quadros certos: {pct(r['acerto_quadros'])}"
              f"{'  (espelhado)' if r['espelhado'] else ''}")
    salvar(pd.DataFrame(linhas), args.saida)
    print(f"\n[RESULTADOS] {os.path.relpath(args.saida, SCRIPT_DIR)}{os.sep}")


if __name__ == "__main__":
    main()
