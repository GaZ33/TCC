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
# --- Janela enviada ---
TAMANHO_JANELA = 120      # maxlen da deque: 120 frames (4 s a 30 fps)

# --- Step: a cada quantos frames NOVOS a janela é reenviada (depois do 1º envio) ---
STEP_ELEVADO = 15         # movimento acelerado   -> reenvia com mais frequência
STEP_NOMINAL = 30         # movimento dentro da faixa
STEP_REDUZIDO = 45        # movimento muito lento -> reenvia com menos frequência

# --- Medida de movimento (a mesma do script movimento_pontas_dedos) ---
JANELA_MOVIMENTO = 10     # frames usados na média do movimento
USAR_Z = False            # False: x e y | True: x, y e z

# --- Faixa de movimento, histerese e confirmação ---
FAIXA_BAIXA = 0.08        # média (mov/ref) ABAIXO disso -> BAIXA_DINAMICA
FAIXA_ALTA = 0.25         # média (mov/ref) ACIMA disso  -> ALTA_DINAMICA
MARGEM_HISTERESE = 0.02   # folga para SAIR de um regime (evita oscilar na borda da faixa)
FRAMES_CONFIRMACAO = 15   # frames seguidos no novo regime antes de o step mudar
MIN_FRAMES_COM_MAO = 0    # só envia a janela se tiver ao menos N frames COM mão (0 = sempre envia, até janela vazia)

# --- ZeroMQ ---
ENDERECO = "tcp://127.0.0.1:5555"

# Índices dos landmarks da mão no MediaPipe
PULSO = 0
BASE_MEDIO = 9
PONTAS = [4, 8, 12, 16, 20]
DIM = 3 if USAR_Z else 2


# =====================================================================
# REGIMES DE MOVIMENTO (renomeie aqui se quiser outros nomes)
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
    Para SAIR de ALTA ou BAIXA é preciso voltar além do limite por uma margem.
    Assim, um valor que oscila em cima do limite não fica trocando de regime."""
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
    """Decide o step. Guarda três coisas:
       - candidato:  regime que a intensidade atual indica (já com histerese)
       - confirmado: regime que de fato está valendo (é ele que define o step)
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
            self.contador = 0              # nada pendente
        elif novo == self.candidato:
            self.contador += 1             # mesmo candidato de antes: soma mais um frame
        else:
            self.contador = 1              # candidato novo: recomeça a contagem
        self.candidato = novo

        if self.contador >= FRAMES_CONFIRMACAO:
            self.confirmado = novo
            self.step = STEP_POR_REGIME[novo]
            self.contador = 0
        return self.step


# =====================================================================
# FUNÇÕES AUXILIARES
# =====================================================================
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
    """Uma linha de dados: frame_id + 21 x (x, y, z, vis) + confiança (mesmo formato do CSV anterior)."""
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


def fmt(valor, casas):
    return "--" if valor is None else f"{valor:.{casas}f}"


def texto(img, s, org, escala, cor):
    """Texto com contorno preto (a fonte do OpenCV não suporta acentos)."""
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0), 4)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, escala, cor, 2)


# =====================================================================
# 1. Modelo do MediaPipe
# =====================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'hand_landmarker.task')

if not os.path.exists(MODEL_PATH):
    print("[DOWNLOAD] O arquivo 'hand_landmarker.task' nao foi encontrado. Baixando automaticamente...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    try:
        urllib.request.urlretrieve(url, MODEL_PATH)
        print("[DOWNLOAD] Concluído com sucesso!")
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
socket_envio.connect(ENDERECO)

# =====================================================================
# 3. Câmera
# =====================================================================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERRO] Não foi possível abrir a câmera.")
    exit(1)

cap.set(cv2.CAP_PROP_FPS, 30)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# =====================================================================
# 4. Estado do programa
# =====================================================================
buffer = deque(maxlen=TAMANHO_JANELA)         # cheia: cada append descarta o elemento mais antigo
historico_mov = deque(maxlen=JANELA_MOVIMENTO)
controle = ControladorStep()
step = controle.step

pontas_anterior = None
frame_id = 0
frames_desde_envio = 0
n_envios = 0
intensidade = None

print("Pressione 'q' para SAIR.")
print(f"Enviando para {ENDERECO} | janela de {TAMANHO_JANELA} frames | step inicial {step}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Erro ao acessar a câmera.")
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    resultado = detector.detect(mp_image)

    mao_detectada = False
    frame_id += 1     # todo frame capturado é numerado, com ou sem mão

    if resultado.hand_landmarks:
        mao_detectada = True
        hand = resultado.hand_landmarks[0]
        hand_score = resultado.handedness[0][0].score

        # --- 1) Mede o movimento das pontas dos dedos ---
        pts = landmarks_para_px(hand, width, height)
        pontas = pts[PONTAS]
        referencia = norma(pts[BASE_MEDIO] - pts[PULSO])

        # Só compara se já existe um frame anterior COM mão
        if pontas_anterior is not None and referencia > 0:
            movimento, mov_norm = calcular_movimento(pontas, pontas_anterior, referencia)
            historico_mov.append(mov_norm)
        pontas_anterior = pontas

        linha = montar_linha(frame_id, hand, hand_score)

        # --- Desenhos só para a tela (nada disso é enviado) ---
        p_pulso = (int(pts[PULSO][0]), int(pts[PULSO][1]))
        p_base = (int(pts[BASE_MEDIO][0]), int(pts[BASE_MEDIO][1]))
        cv2.line(frame, p_pulso, p_base, (0, 255, 255), 2)
        for p in pontas:
            cv2.circle(frame, (int(p[0]), int(p[1])), 7, (0, 0, 255), -1)

    else:
        # Sem mão: o frame é capturado do mesmo jeito.
        # Conta como movimento zero (baixa dinâmica) e vira uma linha de zeros.
        pontas_anterior = None        # não compara a posição de antes com a de depois da ausência
        historico_mov.append(0.0)
        linha = montar_linha_sem_mao(frame_id)

    # --- 2) Atualiza regime e step (só depois de ter JANELA_MOVIMENTO valores) ---
    if len(historico_mov) == JANELA_MOVIMENTO:
        intensidade = sum(historico_mov) / JANELA_MOVIMENTO
        anterior = step
        step = controle.atualizar(intensidade)
        if step != anterior:
            print(f"[STEP] {anterior} -> {step} (regime confirmado: {controle.confirmado.name})")

    # --- 3) Guarda o frame (com ou sem mão) na deque ---
    buffer.append(linha)
    frames_desde_envio += 1

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
            }
            try:
                socket_envio.send_pyobj(pacote, flags=zmq.NOBLOCK)
                n_envios += 1
                print(f"[ENVIO {n_envios}] frames {buffer[0][0]}..{buffer[-1][0]} | com mão: {frames_com_mao}/{TAMANHO_JANELA} | step={step} | {controle.confirmado.name}")
            except zmq.Again:
                print("[AVISO] Fila de envio cheia ou receptor indisponível: janela descartada.")
        else:
            print(f"[PULO] Janela com só {frames_com_mao} frames com mão (mínimo {MIN_FRAMES_COM_MAO}): não enviada.")
        frames_desde_envio = 0

    linhas = [
        f"Mao: {'SIM' if mao_detectada else 'NAO'}",
        f"Intensidade (media {JANELA_MOVIMENTO}): {fmt(intensidade, 4)} | faixa {FAIXA_BAIXA}-{FAIXA_ALTA}",
        f"Candidato:  {controle.candidato.name}",
        f"Confirmado: {controle.confirmado.name} ({controle.contador}/{FRAMES_CONFIRMACAO})",
        f"Step: {step} | Deque: {len(buffer)}/{TAMANHO_JANELA} | Envios: {n_envios}",
        f"Com mao na janela: {sum(1 for lin in buffer if lin[-1] > 0)}/{len(buffer)}",
    ]
    for i, linha in enumerate(linhas):
        texto(frame, linha, (10, 25 + i * 25), 0.55, (255, 255, 255))

    cv2.imshow("Emissor - janela de landmarks", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
socket_envio.close()
contexto.term()
