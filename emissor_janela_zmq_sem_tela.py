"""
emissor_janela_zmq_sem_tela_1.py: começo da cadeia (EMISSOR -> Classificador -> Filtro -> Exibidor).

Captura a câmera (sem abrir janela), extrai os landmarks da mão com o MediaPipe e:
  * envia ao classificador (tcp://127.0.0.1:5555) a janela deslizante de frames, com o lado
    da mão ("maos") e a proporção da imagem ("proporcao") de que as características dependem;
  * envia ao exibidor (tcp://127.0.0.1:5558) a imagem da câmera, com a mão desenhada.
"""

import os
import urllib.request
from collections import deque
from enum import Enum

import cv2
import numpy as np
import mediapipe as mp
import zmq
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =====================================================================
# CONFIGURAÇÕES (ajuste aqui)
# =====================================================================
MOSTRAR_LOG = False       # True: escreve [STEP], [ENVIO], [RESET] no terminal | False: silêncio total

# --- Janela enviada ---
TAMANHO_JANELA = 120      # maxlen da deque: 120 frames (4 s a 30 fps)

# --- Step: a cada quantos frames NOVOS a janela é reenviada (depois do 1º envio) ---
STEP_ELEVADO = 15         # movimento acelerado   -> reenvia com mais frequência
STEP_NOMINAL = 30         # movimento dentro da faixa
STEP_REDUZIDO = 45        # movimento muito lento -> reenvia com menos frequência

# --- Medida de movimento ---
JANELA_MOVIMENTO = 10     # frames usados na média do movimento
USAR_Z = False            # False: x e y | True: x, y e z

# --- Faixa de movimento, histerese e confirmação ---
FAIXA_BAIXA = 0.08        # média (mov/ref) ABAIXO disso -> BAIXA_DINAMICA
FAIXA_ALTA = 0.25         # média (mov/ref) ACIMA disso  -> ALTA_DINAMICA
MARGEM_HISTERESE = 0.02   # folga para SAIR de um regime (evita oscilar na borda da faixa)
FRAMES_CONFIRMACAO = 15   # frames seguidos no novo regime antes de o step mudar
MIN_FRAMES_COM_MAO = 0    # só envia a janela se tiver ao menos N frames COM mão (0 = sempre envia, até janela vazia)

# --- Câmera ---
LARGURA_CAMERA = 640      # mesma resolução das gravações usadas no treino (se a câmera aceitar)
ALTURA_CAMERA = 480

# --- ZeroMQ ---
ENDERECO = "tcp://127.0.0.1:5555"           # janelas para o classificador (PUSH, connect)

# --- Prévia da câmera para o exibidor (o exibidor mostra a imagem; o emissor segue sem janela) ---
ENVIAR_PREVIA = True
ENDERECO_PREVIA = "tcp://127.0.0.1:5558"    # imagens JPEG para o exibidor (PUSH, connect)
PREVIA_A_CADA_N_FRAMES = 2                  # 2 = uma imagem a cada 2 frames (15 por segundo a 30 fps)
QUALIDADE_JPEG = 70

# Índices dos landmarks da mão no MediaPipe
PULSO = 0
BASE_MEDIO = 9
PONTAS = [4, 8, 12, 16, 20]
DIM = 3 if USAR_Z else 2

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # Polegar
    (0, 5), (5, 6), (6, 7), (7, 8),         # Indicador
    (5, 9), (9, 10), (10, 11), (11, 12),    # Médio
    (9, 13), (13, 14), (14, 15), (15, 16),  # Anelar
    (13, 17), (17, 18), (18, 19), (19, 20), # Mínimo
    (0, 17)                                 # Base da palma
]


# =====================================================================
# REGIMES DE MOVIMENTO
# =====================================================================
class Regime(Enum):
    BAIXA_DINAMICA = 1      # movimento abaixo da faixa definida
    DINAMICA_NOMINAL = 2    # movimento dentro da faixa definida
    ALTA_DINAMICA = 3       # movimento acima da faixa definida


STEP_POR_REGIME = {
    Regime.ALTA_DINAMICA: STEP_ELEVADO,
    Regime.DINAMICA_NOMINAL: STEP_NOMINAL,
    Regime.BAIXA_DINAMICA: STEP_REDUZIDO,
}


def classificar_com_histerese(intensidade, regime_anterior):
    """Classifica a intensidade em um regime, com histerese.
    Para ENTRAR em ALTA ou BAIXA é preciso cruzar o limite da faixa.
    Para SAIR de ALTA ou BAIXA é preciso voltar além do limite por uma margem."""
    if regime_anterior == Regime.ALTA_DINAMICA and intensidade >= FAIXA_ALTA - MARGEM_HISTERESE:
        return Regime.ALTA_DINAMICA
    if regime_anterior == Regime.BAIXA_DINAMICA and intensidade <= FAIXA_BAIXA + MARGEM_HISTERESE:
        return Regime.BAIXA_DINAMICA

    if intensidade > FAIXA_ALTA:
        return Regime.ALTA_DINAMICA
    if intensidade < FAIXA_BAIXA:
        return Regime.BAIXA_DINAMICA
    return Regime.DINAMICA_NOMINAL


class ControladorStep:
    """Decide o step. Guarda:
       - candidato:  regime que a intensidade atual indica (já com histerese)
       - confirmado: regime que de fato está valendo (define o step)
       - contador:   há quantos frames seguidos o candidato difere do confirmado
    O step só muda quando o contador chega a FRAMES_CONFIRMACAO."""

    def __init__(self):
        self.reiniciar()

    def reiniciar(self):
        self.candidato = Regime.DINAMICA_NOMINAL
        self.confirmado = Regime.DINAMICA_NOMINAL
        self.contador = 0
        self.step = STEP_POR_REGIME[self.confirmado]

    def atualizar(self, intensidade):
        novo = classificar_com_histerese(intensidade, self.candidato)

        if novo == self.confirmado:
            self.contador = 0
        elif novo == self.candidato:
            self.contador += 1
        else:
            self.contador = 1
        self.candidato = novo

        if self.contador >= FRAMES_CONFIRMACAO:
            self.confirmado = novo
            self.step = STEP_POR_REGIME[novo]
            self.contador = 0
        return self.step


# =====================================================================
# FUNÇÕES AUXILIARES
# =====================================================================
def log(mensagem):
    """Só escreve no terminal se MOSTRAR_LOG estiver ligado."""
    if MOSTRAR_LOG:
        print(mensagem)


def norma(v):
    """Raiz da soma dos quadrados dos elementos (norma euclidiana)."""
    v = np.asarray(v, dtype=float)
    return float(np.sqrt(np.sum(v ** 2)))


def landmarks_para_px(hand, largura, altura):
    """Landmarks (0 a 1) -> pixels. Retorna array de formato (21, DIM)."""
    pts = np.array([[lm.x * largura, lm.y * altura, lm.z * largura] for lm in hand])
    return pts[:, :DIM]


def calcular_movimento(pontas, pontas_anterior, referencia):
    """Deslocamento das 5 pontas entre dois frames: (px, px / referência)."""
    movimento = norma(pontas - pontas_anterior)
    return movimento, movimento / referencia


def montar_linha(frame_id, hand, hand_score):
    """frame_id + 21 x (x, y, z, vis) + confiança (mesmo formato do CSV anterior)."""
    linha = [frame_id]
    for lm in hand:
        vis = getattr(lm, 'visibility', None)
        if vis is None:           # no Hand Landmarker a visibilidade costuma vir vazia
            vis = 0.0
        linha.extend([lm.x, lm.y, lm.z, vis])
    linha.append(hand_score)
    return linha


def montar_linha_sem_mao(frame_id):
    """Frame sem mão: coordenadas zeradas e hand_confidence = 0 (é isso que marca a ausência).
    Tem o mesmo tamanho de uma linha normal: 1 + 21*4 + 1 = 86 valores."""
    return [frame_id] + [0.0] * (21 * 4 + 1)


def desenhar_esqueleto(imagem, pontos_px):
    """Desenha a mão detectada (só as linhas verdes, sem os pontos) direto na imagem."""
    for a, b in HAND_CONNECTIONS:
        cv2.line(imagem, pontos_px[a], pontos_px[b], (0, 255, 0), 2)


def enviar_previa(socket_previa, imagem, pontos_px):
    """Manda a imagem (com a mão desenhada, se houver) ao exibidor em JPEG, sem nunca travar."""
    if pontos_px is not None:
        desenhar_esqueleto(imagem, pontos_px)
    ok, jpeg = cv2.imencode(".jpg", imagem, [cv2.IMWRITE_JPEG_QUALITY, QUALIDADE_JPEG])
    if not ok:
        return
    try:
        socket_previa.send(jpeg.tobytes(), flags=zmq.NOBLOCK)
    except zmq.Again:
        pass      # exibidor fechado ou lento: uma imagem velha não interessa


# =====================================================================
# 1. Modelo do MediaPipe
# =====================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'hand_landmarker.task')

if not os.path.exists(MODEL_PATH):
    log("[DOWNLOAD] Baixando 'hand_landmarker.task'...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    try:
        urllib.request.urlretrieve(url, MODEL_PATH)
        log("[DOWNLOAD] Concluído.")
    except Exception as e:
        print(f"[ERRO] Falha ao baixar o modelo: {e}")
        exit(1)

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    running_mode=vision.RunningMode.IMAGE
)
detector = vision.HandLandmarker.create_from_options(options)

# Cabeçalho das colunas (enviado junto com os dados para o receptor montar o CSV)
columns = ['frame_id']
for i in range(21):
    columns.extend([f'lm_{i}_x', f'lm_{i}_y', f'lm_{i}_z', f'lm_{i}_vis'])
columns.append('hand_confidence')

# =====================================================================
# 2. ZeroMQ (PUSH: este script ENVIA; o receptor faz bind, este faz connect)
# =====================================================================
contexto = zmq.Context()
socket_envio = contexto.socket(zmq.PUSH)
socket_envio.setsockopt(zmq.LINGER, 1000)     # ao encerrar, espera até 1 s para entregar o que sobrou
socket_envio.setsockopt(zmq.SNDHWM, 4)        # fila curta: janela velha não interessa ao classificador
socket_envio.setsockopt(zmq.IMMEDIATE, 1)     # sem classificador conectado, descarta em vez de acumular
socket_envio.connect(ENDERECO)

socket_previa = None
if ENVIAR_PREVIA:
    socket_previa = contexto.socket(zmq.PUSH)
    socket_previa.setsockopt(zmq.LINGER, 0)
    socket_previa.setsockopt(zmq.SNDHWM, 2)
    socket_previa.setsockopt(zmq.IMMEDIATE, 1)
    socket_previa.connect(ENDERECO_PREVIA)

# =====================================================================
# 3. Câmera
# =====================================================================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERRO] Não foi possível abrir a câmera.")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, LARGURA_CAMERA)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTURA_CAMERA)
cap.set(cv2.CAP_PROP_FPS, 30)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))     # resolução que a câmera realmente aceitou
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
proporcao = width / height if height else LARGURA_CAMERA / ALTURA_CAMERA   # o classificador usa isto
if (width, height) != (LARGURA_CAMERA, ALTURA_CAMERA):
    print(f"[AVISO] A câmera abriu em {width}x{height}, e o treino usou {LARGURA_CAMERA}x{ALTURA_CAMERA}. "
          "A proporção real é enviada ao classificador.")

# =====================================================================
# 4. Estado do programa
# =====================================================================
buffer = deque(maxlen=TAMANHO_JANELA)         # cheia: cada append descarta o elemento mais antigo
buffer_maos = deque(maxlen=TAMANHO_JANELA)    # lado da mão de cada frame ("Left", "Right" ou None)
historico_mov = deque(maxlen=JANELA_MOVIMENTO)
controle = ControladorStep()
step = controle.step

pontas_anterior = None     # None = ainda não existe frame anterior para comparar
frame_id = 0
frames_desde_envio = 0
n_envios = 0

log("Capturando sem janela. Pressione Ctrl+C no terminal para sair.")

# Sem janela não existe cv2.waitKey (ele só lê teclas de uma janela do OpenCV),
# então o programa roda até você apertar Ctrl+C.
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERRO] Falha ao ler a câmera.")
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        resultado = detector.detect(mp_image)

        frame_id += 1     # todo frame capturado é numerado, com ou sem mão
        pontos_px = None  # só preenchido quando há mão (usado no desenho da prévia)

        if resultado.hand_landmarks:
            hand = resultado.hand_landmarks[0]
            hand_score = resultado.handedness[0][0].score
            lado = resultado.handedness[0][0].category_name   # "Left"/"Right" (frame espelhado)

            # --- 1) Mede o movimento das pontas dos dedos ---
            pts = landmarks_para_px(hand, width, height)
            pontos_px = [(int(x), int(y)) for x, y in pts[:, :2]]
            pontas = pts[PONTAS]
            referencia = norma(pts[BASE_MEDIO] - pts[PULSO])

            # Só compara se já existe um frame anterior COM mão
            if pontas_anterior is not None and referencia > 0:
                movimento, mov_norm = calcular_movimento(pontas, pontas_anterior, referencia)
                historico_mov.append(mov_norm)
            pontas_anterior = pontas

            linha = montar_linha(frame_id, hand, hand_score)

        else:
            # Sem mão: o frame é capturado do mesmo jeito.
            # Conta como movimento zero (baixa dinâmica) e vira uma linha de zeros.
            lado = None
            pontas_anterior = None        # não compara a posição de antes com a de depois da ausência
            historico_mov.append(0.0)
            linha = montar_linha_sem_mao(frame_id)

        # --- 2) Atualiza regime e step (só depois de ter JANELA_MOVIMENTO valores) ---
        if len(historico_mov) == JANELA_MOVIMENTO:
            intensidade = sum(historico_mov) / JANELA_MOVIMENTO
            anterior = step
            step = controle.atualizar(intensidade)
            if step != anterior:
                log(f"[STEP] {anterior} -> {step} (regime confirmado: {controle.confirmado.name})")

        # --- 3) Guarda o frame (com ou sem mão) na deque ---
        buffer.append(linha)
        buffer_maos.append(lado)
        frames_desde_envio += 1

        # --- 3b) Prévia para o exibidor (não interfere nos dados enviados ao classificador) ---
        if socket_previa is not None and frame_id % PREVIA_A_CADA_N_FRAMES == 0:
            enviar_previa(socket_previa, frame, pontos_px)

        # --- 4) Envia: 1º envio quando a deque enche; depois a cada `step` frames ---
        if len(buffer) == TAMANHO_JANELA and frames_desde_envio >= step:
            frames_com_mao = sum(1 for lin in buffer if lin[-1] > 0)   # hand_confidence > 0 = o frame tinha mão
            if frames_com_mao >= MIN_FRAMES_COM_MAO:
                pacote = {
                    "sequencia": n_envios + 1,
                    "step": step,
                    "regime": controle.confirmado.name,
                    "frames_com_mao": frames_com_mao,
                    "colunas": columns,
                    "amostras": list(buffer),
                    "maos": list(buffer_maos),   # um por linha de "amostras": o classificador precisa do lado
                    "proporcao": proporcao,      # largura / altura da imagem
                }
                try:
                    socket_envio.send_pyobj(pacote, flags=zmq.NOBLOCK)
                    n_envios += 1
                    log(f"[ENVIO {n_envios}] frames {buffer[0][0]}..{buffer[-1][0]} | com mão: {frames_com_mao}/{TAMANHO_JANELA} | step={step} | {controle.confirmado.name}")
                except zmq.Again:
                    log("[AVISO] Fila de envio cheia ou receptor indisponível: janela descartada.")
            else:
                log(f"[PULO] Janela com só {frames_com_mao} frames com mão (mínimo {MIN_FRAMES_COM_MAO}): não enviada.")
            frames_desde_envio = 0

except KeyboardInterrupt:
    pass
finally:
    cap.release()
    socket_envio.close()
    if socket_previa is not None:
        socket_previa.close()
    contexto.term()