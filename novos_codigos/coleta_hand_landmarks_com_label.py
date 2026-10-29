"""
Coleta de landmarks da mão (MediaPipe) - VERSÃO COM LABEL

Fluxo:
  1. Pressione uma letra (A-Z): define "qual tecla será gravada".
  2. Pressione TAB: alterna a mão da gravação (direita/esquerda).
  3. Pressione ESPAÇO: grava TOTAL_COLETAS frames, com ou sem mão na tela.
  4. Ao final salva "letra <x> <mao>.csv" e "letra <x> <mao>.mp4" na pasta do script.
     Se já existir uma gravação com esse nome, usa "letra <x> <mao>_2", "_3"...
     (nunca sobrescreve uma gravação anterior).
  ESC: sai do programa.

O CSV recebe a coluna 'label' (a letra gravada), pronta para treinar
modelos supervisionados como Random Forest.
"""

import os
import urllib.request
import cv2
import mediapipe as mp
import pandas as pd
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math

# 1. Garante que o modelo 'hand_landmarker.task' esteja na mesma pasta do script
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

# 2. Configuração do modelo MediaPipe Hand Landmarker
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    running_mode=vision.RunningMode.IMAGE
)

detector = vision.HandLandmarker.create_from_options(options)

# 3. Mapeamento das 21 conexões da mão
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Polegar
    (0, 5), (5, 6), (6, 7), (7, 8),        # Indicador
    (5, 9), (9, 10), (10, 11), (11, 12),   # Médio
    (9, 13), (13, 14), (14, 15), (15, 16), # Anelar
    (13, 17), (17, 18), (18, 19), (19, 20),# Mínimo
    (0, 17)                                # Base da palma
]

# 4. Cabeçalho do CSV
columns = ['frame_id']

for i in range(21):
    columns.extend([
        f'lm_{i}_x',
        f'lm_{i}_y',
        f'lm_{i}_z',
        f'lm_{i}_vis'
    ])

columns.append('hand_confidence')
columns.append('label')  # coluna-alvo (y): a letra gravada

# 5. Inicialização da câmera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERRO] Não foi possível abrir a câmera.")
    exit(1)

fps = 30
cap.set(cv2.CAP_PROP_FPS, fps)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

TOTAL_COLETAS = 90

# Valor usado nos landmarks dos frames em que NÃO há mão na tela
# (esses frames também têm hand_confidence = 0.0)
VALOR_SEM_MAO = 0.0

coletas = []
gravando = False
video_writer = None

# Tecla (letra) escolhida e nomes dos arquivos da gravação atual
letra_atual = None
nome_csv = None

# Mão usada na gravação ("direita" ou "esquerda"); vai no nome dos arquivos.
# TAB alterna entre as duas.
mao_atual = "direita"
nome_video = None

# Mão que o MediaPipe vê. No frame espelhado ele chama a mão DIREITA do usuário
# de 'Left' (mesma regra de caracteristicas_mao.mao_esquerda, na raiz do projeto).
# Serve para avisar quando a mão na tela não é a escolhida no TAB.
MAO_DO_MEDIAPIPE = {"Left": "direita", "Right": "esquerda"}

# Contagem da gravação atual: frames com mão e frames com a mão diferente da escolhida
frames_com_mao_gravacao = 0
frames_mao_divergente = 0

# Controle de Feedback Visual na Tela
feedback_timer = 0
MENSAGEM_FEEDBACK = ""


# =============================================================
# FUNÇÃO PARA CALCULAR DISTÂNCIA EM PIXELS
# =============================================================

def distancia_pixels(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    return math.sqrt(dx * dx + dy * dy)


# =============================================================
# FUNÇÃO PARA ESCOLHER NOMES DE ARQUIVO AINDA NÃO USADOS
# =============================================================

def nomes_livres(nome_base):
    """Devolve (caminho_csv, caminho_video) que ainda NÃO existem.
    Tenta 'letra a direita'; se o CSV ou o vídeo já existir, tenta
    'letra a direita_2', 'letra a direita_3'... e assim por diante.
    Assim uma gravação nova nunca apaga uma gravação anterior."""

    n = 1

    while True:

        nome = nome_base if n == 1 else f"{nome_base}_{n}"

        caminho_csv = os.path.join(SCRIPT_DIR, f"{nome}.csv")
        caminho_video = os.path.join(SCRIPT_DIR, f"{nome}.mp4")

        if not os.path.exists(caminho_csv) and not os.path.exists(caminho_video):
            return caminho_csv, caminho_video

        n += 1


print("=" * 60)
print("INSTRUÇÕES (versão COM label):")
print(" - Pressione uma LETRA ou NÚMERO (A-Z, 0-9) para escolher qual tecla será gravada.")
print(" - Pressione TAB para trocar a mão da gravação (direita/esquerda).")
print(f" - Pressione ESPAÇO para gravar {TOTAL_COLETAS} amostras (com ou sem mão).")
print(" - Pressione ESC para SAIR.")
print("=" * 60)


while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        print("Erro ao acessar a câmera.")
        break

    # =========================================================
    # FRAME ORIGINAL
    # =========================================================

    frame_original = cv2.flip(frame, 1)

    frame_rgb = cv2.cvtColor(
        frame_original,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )

    detection_result = detector.detect(mp_image)

    # =========================================================
    # FRAME DE DESENHO
    # =========================================================

    frame_desenho = frame_original.copy()

    mao_detectada = False
    hand = None
    hand_score = 0.0
    mao_vista = None  # "direita" / "esquerda" segundo o MediaPipe

    if detection_result.hand_landmarks:

        mao_detectada = True

        hand = detection_result.hand_landmarks[0]

        hand_score = detection_result.handedness[0][0].score

        mao_vista = MAO_DO_MEDIAPIPE.get(
            detection_result.handedness[0][0].category_name
        )

        # Converte os landmarks normalizados para pixels
        points_px = [
            (
                int(lm.x * width),
                int(lm.y * height)
            )
            for lm in hand
        ]

        # =====================================================
        # CÁLCULO DO TAMANHO DA MÃO EM PIXELS
        # =====================================================

        d_05 = distancia_pixels(
            points_px[0],
            points_px[5]
        )

        d_09 = distancia_pixels(
            points_px[0],
            points_px[9]
        )

        d_013 = distancia_pixels(
            points_px[0],
            points_px[13]
        )

        d_017 = distancia_pixels(
            points_px[0],
            points_px[17]
        )

        tamanho_mao_px = (
            d_05 +
            d_09 +
            d_013 +
            d_017
        ) / 4

        # =====================================================
        # DESENHO DO ESQUELETO
        # =====================================================

        for connection in HAND_CONNECTIONS:

            cv2.line(
                frame_desenho,
                points_px[connection[0]],
                points_px[connection[1]],
                (0, 255, 0),
                2
            )

        # Desenha os pontos dos landmarks
        for px in points_px:

            cv2.circle(
                frame_desenho,
                px,
                5,
                (0, 0, 255),
                -1
            )

        # =====================================================
        # FEEDBACK DO TAMANHO DA MÃO
        #
        # IMPORTANTE:
        # Este texto é desenhado SOMENTE em frame_desenho.
        # Portanto NÃO será gravado no vídeo.
        # =====================================================

        cv2.putText(
            frame_desenho,
            f"Tamanho da mao: {tamanho_mao_px:.1f} px",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        # Mostra também as quatro medidas individuais
        cv2.putText(
            frame_desenho,
            f"0-5: {d_05:.1f} | 0-9: {d_09:.1f}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame_desenho,
            f"0-13: {d_013:.1f} | 0-17: {d_017:.1f}",
            (20, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    # =========================================================
    # LÓGICA DE CAPTURA E GRAVAÇÃO
    #
    # Fica FORA do bloco de detecção: a cada frame, depois do
    # ESPAÇO, uma amostra é gravada com ou sem mão na tela.
    # =========================================================

    if gravando:

        frame_actual_id = len(coletas) + 1

        # Dados do CSV
        vetor_coleta = [frame_actual_id]

        if mao_detectada:

            for lm in hand:

                vis = getattr(
                    lm,
                    'visibility',
                    None
                )

                if vis is None:
                    vis = 0.0

                vetor_coleta.extend([
                    lm.x,
                    lm.y,
                    lm.z,
                    vis
                ])

            vetor_coleta.append(hand_score)

            frames_com_mao_gravacao += 1
            if mao_vista is not None and mao_vista != mao_atual:
                frames_mao_divergente += 1

        else:

            # Sem mão: 21 landmarks x 4 valores + confiança zerada
            vetor_coleta.extend([VALOR_SEM_MAO] * (21 * 4))
            vetor_coleta.append(0.0)

        vetor_coleta.append(letra_atual)  # label

        coletas.append(vetor_coleta)

        # =================================================
        # GRAVA SOMENTE O FRAME ORIGINAL
        # =================================================

        video_writer.write(frame_original)

        # Indicador REC SOMENTE NA TELA
        cv2.putText(
            frame_desenho,
            f"REC [{letra_atual.upper()}] - Amostra: {frame_actual_id}/{TOTAL_COLETAS}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )

        cv2.circle(
            frame_desenho,
            (15, 42),
            8,
            (0, 0, 255),
            -1
        )

        # =================================================
        # FINAL DA COLETA
        # =================================================

        if len(coletas) == TOTAL_COLETAS:

            gravando = False

            video_writer.release()
            video_writer = None

            print("\n" + "=" * 60)
            print("[PROCESSANDO] Gravando planilha no disco...")

            # Salva CSV
            df = pd.DataFrame(
                coletas,
                columns=columns
            )

            df.to_csv(
                nome_csv,
                index=False
            )

            MENSAGEM_FEEDBACK = (
                f"SALVO: {os.path.basename(nome_csv)}"
            )

            feedback_timer = 90

            print(
                f"[SUCESSO] Planilha CSV salva em: {nome_csv}"
            )

            print(
                f"[SUCESSO] Vídeo MP4 salvo em: {nome_video}"
            )

            # A maioria dos frames com mão mostrou a OUTRA mão: nome provavelmente errado
            if frames_mao_divergente > frames_com_mao_gravacao / 2:
                outra_mao = "esquerda" if mao_atual == "direita" else "direita"
                print(
                    f"[AVISO] Em {frames_mao_divergente} de {frames_com_mao_gravacao} frames "
                    f"com mão, o MediaPipe viu a mão {outra_mao}, mas o arquivo "
                    f"'{os.path.basename(nome_csv)}' diz {mao_atual}. "
                    f"Confira o TAB e, se for o caso, grave de novo ou renomeie."
                )

            print("=" * 60 + "\n")

    # =========================================================
    # FEEDBACK VISUAL
    # =========================================================

    if feedback_timer > 0:

        cv2.rectangle(
            frame_desenho,
            (0, 0),
            (width, 60),
            (0, 180, 0),
            -1
        )

        cv2.putText(
            frame_desenho,
            f"[OK] {MENSAGEM_FEEDBACK}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        feedback_timer -= 1

    elif not gravando:

        if letra_atual is None:
            prompt_txt = "Sera gravado qual tecla? (pressione uma letra)"
        else:
            prompt_txt = f"Sera gravado qual tecla? -> {letra_atual.upper()}"

        cv2.putText(
            frame_desenho,
            prompt_txt,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    # Mão escolhida para a gravação (sempre visível, em amarelo)
    cv2.putText(
        frame_desenho,
        f"Mao da gravacao: {mao_atual.upper()} (TAB troca)",
        (20, height - 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    # Mão que o MediaPipe está vendo (vermelho quando diverge da escolhida)
    if mao_vista is not None:

        if mao_vista == mao_atual:
            texto_mao_vista = f"Mao detectada: {mao_vista.upper()}"
            cor_mao_vista = (0, 255, 0)
        else:
            texto_mao_vista = f"Mao detectada: {mao_vista.upper()} (DIFERENTE!)"
            cor_mao_vista = (0, 0, 255)
            cv2.rectangle(
                frame_desenho,
                (10, height - 100),
                (500, height - 68),
                (255, 255, 255),
                -1
            )

        cv2.putText(
            frame_desenho,
            texto_mao_vista,
            (20, height - 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            cor_mao_vista,
            2
        )

    # Dicas fixas na parte de baixo da tela
    cv2.putText(
        frame_desenho,
        f"ESPACO: gravar | ESC: sair | Mao na tela: {'SIM' if mao_detectada else 'NAO'}",
        (20, height - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    # =========================================================
    # MOSTRA SOMENTE O FRAME DE DESENHO
    # =========================================================

    cv2.imshow(
        "Coleta de Landmarks - MediaPipe",
        frame_desenho
    )

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC

        break

    if not gravando:

        if (ord('a') <= key <= ord('z') or 
            ord('A') <= key <= ord('Z') or 
            ord('0') <= key <= ord('9')):

            # Escolhe (ou troca) a tecla (letra ou número) que será gravada
            letra_atual = chr(key).lower()

            feedback_timer = 0

        elif key == 9:  # TAB

            # Alterna a mão da gravação (bloqueado durante a gravação)
            if mao_atual == "direita":
                mao_atual = "esquerda"
            else:
                mao_atual = "direita"

        elif key == 32:  # ESPAÇO

            if letra_atual is None:

                print("[AVISO] Escolha uma letra (A-Z) antes de pressionar ESPAÇO.")

            else:

                # Nomes ainda não usados: nunca sobrescreve outra gravação
                nome_csv, nome_video = nomes_livres(
                    f"letra {letra_atual} {mao_atual}"
                )

                print(
                    f"\n[INÍCIO] Gravando letra '{letra_atual}' (mão {mao_atual}): "
                    f"{TOTAL_COLETAS} amostras -> {os.path.basename(nome_csv)}"
                )

                coletas = []

                frames_com_mao_gravacao = 0
                frames_mao_divergente = 0

                gravando = True

                feedback_timer = 0

                fourcc = cv2.VideoWriter_fourcc(
                    *'mp4v'
                )

                video_writer = cv2.VideoWriter(
                    nome_video,
                    fourcc,
                    fps,
                    (width, height)
                )


# =============================================================
# LIBERAÇÃO DOS RECURSOS
# =============================================================

cap.release()

if video_writer is not None:
    video_writer.release()

cv2.destroyAllWindows()
