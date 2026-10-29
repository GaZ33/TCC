"""
Classificador ao vivo: mostra na tela a letra que a mão está fazendo.

Uso (de dentro da pasta do projeto):
  .venv\\Scripts\\python classificar_ao_vivo.py
  .venv\\Scripts\\python classificar_ao_vivo.py --video "novos_codigos\\letra a direita.mp4"

Teclas (clique na janela antes):
  TAB: troca a mão -> automático, direita fixa, esquerda fixa
  ESC: sai

A letra mostrada é a mais votada entre os últimos JANELA_VOTOS frames com mão,
e só aparece com pelo menos MINIMO_VOTOS votos. O modelo vem de
treinar_classificador.py. O acerto real com uma pessoa nova é o da avaliação
"pessoa de fora" do treino, não o que aparece aqui com quem gravou os dados.
"""

import argparse
import collections
import os
import time
import urllib.request

import cv2
import joblib
import mediapipe as mp
import numpy as np
import sklearn
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import caracteristicas_mao

# =====================================================================
# CONFIGURAÇÕES
# =====================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CAMINHO_MODELO = os.path.join(SCRIPT_DIR, "modelos", "random_forest_letras.joblib")
MODEL_PATH = os.path.join(SCRIPT_DIR, "hand_landmarker.task")
URL_MODEL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

JANELA_VOTOS = 10        # quantos frames com mão votam na letra
MINIMO_VOTOS = 6         # votos mínimos para mostrar a letra
TOLERANCIA_SEM_MAO = 5   # frames seguidos sem mão até descartar os votos
N_MAIS_PROVAVEIS = 3     # letras mais prováveis do frame atual mostradas na tela

# Ordem do TAB
MODOS_MAO = ["auto", "direita", "esquerda"]

NOME_JANELA = "Classificador ao vivo - Libras"

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Polegar
    (0, 5), (5, 6), (6, 7), (7, 8),        # Indicador
    (5, 9), (9, 10), (10, 11), (11, 12),   # Médio
    (9, 13), (13, 14), (14, 15), (15, 16), # Anelar
    (13, 17), (17, 18), (18, 19), (19, 20),# Mínimo
    (0, 17)                                # Base da palma
]


class ErroModelo(Exception):
    """O modelo salvo não existe ou não serve para este classificador."""


def carregar_modelo(caminho=CAMINHO_MODELO):
    """Carrega o pacote salvo por treinar_classificador.py e confere se ele é compatível."""
    if not os.path.exists(caminho):
        raise ErroModelo(f"Modelo não encontrado em {caminho}. "
                         "Treine primeiro: .venv\\Scripts\\python treinar_classificador.py")

    pacote = joblib.load(caminho)
    if not isinstance(pacote, dict) or "modelo" not in pacote:
        raise ErroModelo(f"{caminho} não foi salvo por treinar_classificador.py.")
    if list(pacote.get("caracteristicas", [])) != caracteristicas_mao.NOMES:
        raise ErroModelo("O modelo foi treinado com outras características da mão. "
                         "Treine de novo: .venv\\Scripts\\python treinar_classificador.py")

    versao = pacote.get("versao_sklearn")
    if versao != sklearn.__version__:
        print(f"[AVISO] Modelo treinado com scikit-learn {versao}, e este é o {sklearn.__version__}. "
              "Se as previsões parecerem estranhas, treine de novo.")

    # Para prever 1 frame por vez, dividir as árvores entre processos custa mais do que economiza
    pacote["modelo"].n_jobs = 1
    return pacote


class VotacaoLetra:
    """Estabiliza a letra: voto entre as previsões dos últimos frames com mão."""

    def __init__(self, janela=JANELA_VOTOS, minimo=MINIMO_VOTOS, tolerancia=TOLERANCIA_SEM_MAO):
        self.votos = collections.deque(maxlen=janela)
        self.minimo = minimo
        self.tolerancia = tolerancia
        self.frames_sem_mao = tolerancia  # começa como "sem mão": nenhuma mão vista ainda

    def com_mao(self, letra):
        self.frames_sem_mao = 0
        self.votos.append(letra)

    def sem_mao(self):
        # Frame sem mão não vota; uma falha curta de detecção não apaga a letra
        self.frames_sem_mao += 1
        if self.frames_sem_mao >= self.tolerancia:
            self.votos.clear()

    @property
    def mao_ausente(self):
        return self.frames_sem_mao >= self.tolerancia

    def resultado(self):
        """(letra, votos) da mais votada; letra é None quando está indeciso."""
        if not self.votos:
            return None, 0
        letra, votos = collections.Counter(self.votos).most_common(1)[0]
        return (letra if votos >= self.minimo else None), votos


def proximo_modo(modo):
    return MODOS_MAO[(MODOS_MAO.index(modo) + 1) % len(MODOS_MAO)]


def mao_em_uso(modo, nome_mediapipe):
    """True se o frame deve ser tratado como mão esquerda."""
    if modo == "auto":
        return caracteristicas_mao.mao_esquerda(nome_mediapipe)
    return modo == "esquerda"


def classificar_frame(pacote, pontos, esquerda, proporcao):
    """pontos: (21, 3) com x, y, z do MediaPipe.
    Retorna [(letra, probabilidade), ...] das N_MAIS_PROVAVEIS, da mais provável para a menos."""
    X = caracteristicas_mao.extrair(np.asarray(pontos, dtype=float)[None], [esquerda], proporcao)
    modelo = pacote["modelo"]
    prob = modelo.predict_proba(X)[0]
    ordem = np.argsort(prob)[::-1][:N_MAIS_PROVAVEIS]
    return [(str(modelo.classes_[i]), float(prob[i])) for i in ordem]


# =====================================================================
# CÂMERA / VÍDEO E MEDIAPIPE
# =====================================================================

def abrir_fonte(video=None):
    """Retorna (captura, espelhar). Os vídeos do coletor já foram gravados espelhados."""
    if video:
        if not os.path.exists(video):
            raise SystemExit(f"[ERRO] Vídeo não encontrado: {video}")
        captura = cv2.VideoCapture(video)
        if not captura.isOpened():
            raise SystemExit(f"[ERRO] Não foi possível abrir o vídeo: {video}")
        return captura, False

    captura = cv2.VideoCapture(0)
    if not captura.isOpened():
        raise SystemExit("[ERRO] Não foi possível abrir a câmera. "
                         "Se houver mais de uma, troque o 0 em cv2.VideoCapture(0).")
    # Mesma resolução das gravações (se a câmera aceitar)
    captura.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    captura.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return captura, True


def criar_detector():
    if not os.path.exists(MODEL_PATH):
        print("[DOWNLOAD] Baixando 'hand_landmarker.task'...")
        try:
            urllib.request.urlretrieve(URL_MODEL, MODEL_PATH)
        except Exception as e:
            raise SystemExit(f"[ERRO] Falha ao baixar o modelo do MediaPipe: {e}")

    # Modo IMAGE, igual ao coletor: o modelo aprendeu com landmarks desse modo
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        num_hands=1,
        running_mode=vision.RunningMode.IMAGE
    )
    return vision.HandLandmarker.create_from_options(options)


# =====================================================================
# TELA
# =====================================================================

def escrever(img, texto, pos, escala, cor, espessura=2):
    cv2.putText(img, texto, pos, cv2.FONT_HERSHEY_SIMPLEX, escala, cor, espessura, cv2.LINE_AA)


def desenhar(img, pontos_px, votacao, mais_provaveis, modo, esquerda, fps):
    altura, largura = img.shape[:2]

    if pontos_px:
        for a, b in HAND_CONNECTIONS:
            cv2.line(img, pontos_px[a], pontos_px[b], (0, 255, 0), 2)
        for px in pontos_px:
            cv2.circle(img, px, 4, (0, 0, 255), -1)

    # Fundos escuros semitransparentes para o texto ficar legível
    fundo = img.copy()
    cv2.rectangle(fundo, (0, 0), (190, 150), (0, 0, 0), -1)
    cv2.rectangle(fundo, (0, altura - 40), (largura, altura), (0, 0, 0), -1)
    if mais_provaveis:
        cv2.rectangle(fundo, (largura - 210, 0), (largura, 30 + 32 * len(mais_provaveis)), (0, 0, 0), -1)
    cv2.addWeighted(fundo, 0.55, img, 0.45, 0, img)

    # Letra estável
    letra, votos = votacao.resultado()
    if votacao.mao_ausente:
        escrever(img, "sem mao", (15, 85), 1.1, (180, 180, 180))
    elif letra is None:
        escrever(img, "?", (55, 110), 3.5, (0, 200, 255), 7)
        escrever(img, f"indeciso ({votos}/{votacao.votos.maxlen})", (10, 140), 0.55, (0, 200, 255))
    else:
        escrever(img, letra.upper(), (40, 110), 3.5, (0, 255, 0), 8)
        escrever(img, f"{votos}/{votacao.votos.maxlen} votos", (40, 140), 0.6, (0, 255, 0))

    # Letras mais prováveis do frame atual
    if mais_provaveis:
        escrever(img, "frame atual:", (largura - 200, 22), 0.55, (255, 255, 255), 1)
        for i, (classe, prob) in enumerate(mais_provaveis):
            y = 50 + 32 * i
            escrever(img, f"{classe} {prob:4.0%}", (largura - 200, y), 0.7, (255, 255, 255))
            cv2.rectangle(img, (largura - 95, y - 15), (largura - 95 + int(85 * prob), y - 1), (0, 200, 255), -1)

    lado = "ESQUERDA" if esquerda else "DIREITA"
    origem = "auto" if modo == "auto" else "fixa"
    texto_mao = f"Mao: {lado} ({origem})" if esquerda is not None else f"Mao: - ({origem})"
    escrever(img, f"{texto_mao} | TAB troca | ESC sai | {fps:4.1f} fps", (10, altura - 13), 0.55, (255, 255, 255), 1)


# =====================================================================
# LAÇO PRINCIPAL
# =====================================================================

def main(argv=None):
    parser = argparse.ArgumentParser(description="Classifica letras do alfabeto manual ao vivo.")
    parser.add_argument("--video", help="processa um .mp4 gravado pelo coletor em vez da câmera")
    parser.add_argument("--modelo", default=CAMINHO_MODELO, help="arquivo .joblib salvo pelo treino")
    args = parser.parse_args(argv)

    try:
        pacote = carregar_modelo(args.modelo)
    except ErroModelo as e:
        raise SystemExit(f"[ERRO] {e}")
    print(f"[MODELO] {len(pacote['classes'])} classes: {' '.join(pacote['classes'])}")

    captura, espelhar = abrir_fonte(args.video)
    detector = criar_detector()

    votacao = VotacaoLetra()
    modo = "auto"
    fps = 0.0
    anterior = time.perf_counter()
    letras_mostradas = collections.Counter()

    print("TAB: troca a mão (automático / direita fixa / esquerda fixa) | ESC: sai")
    try:
        while True:
            ok, frame = captura.read()
            if not ok:
                if args.video:
                    break
                print("[ERRO] A câmera parou de entregar frames.")
                break
            if espelhar:
                frame = cv2.flip(frame, 1)

            altura, largura = frame.shape[:2]
            imagem = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            resultado = detector.detect(imagem)

            pontos_px = None
            mais_provaveis = None
            esquerda = None
            if resultado.hand_landmarks:
                mao = resultado.hand_landmarks[0]
                pontos = [(lm.x, lm.y, lm.z) for lm in mao]
                pontos_px = [(int(x * largura), int(y * altura)) for x, y, _ in pontos]
                esquerda = mao_em_uso(modo, resultado.handedness[0][0].category_name)
                mais_provaveis = classificar_frame(pacote, pontos, esquerda, largura / altura)
                votacao.com_mao(mais_provaveis[0][0])
            else:
                votacao.sem_mao()

            letra, _ = votacao.resultado()
            if letra is not None and not votacao.mao_ausente:
                letras_mostradas[letra] += 1

            agora = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(agora - anterior, 1e-6) if fps else 1 / max(agora - anterior, 1e-6)
            anterior = agora

            desenhar(frame, pontos_px, votacao, mais_provaveis, modo, esquerda, fps)
            cv2.imshow(NOME_JANELA, frame)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla == 27:  # ESC
                break
            if tecla == 9:   # TAB
                modo = proximo_modo(modo)
                print(f"[MÃO] modo: {modo}")
            if cv2.getWindowProperty(NOME_JANELA, cv2.WND_PROP_VISIBLE) < 1:  # fechou no X
                break
    finally:
        captura.release()
        cv2.destroyAllWindows()

    if args.video:
        if letras_mostradas:
            letra, frames = letras_mostradas.most_common(1)[0]
            print(f"letra mais mostrada: {letra} ({frames} frames)")
        else:
            print("nenhuma letra foi mostrada")
    return letras_mostradas


if __name__ == "__main__":
    main()
