"""
Características da mão usadas pelo classificador (no treino e, depois, ao vivo).

Recebe os 21 landmarks do MediaPipe (x, y, z normalizados pela imagem) e devolve
35 números que NÃO dependem de onde a mão está na tela, do tamanho dela na
imagem nem de qual mão (direita/esquerda) fez o sinal:
  - 19 ângulos: dobra de cada articulação dos dedos e abertura entre dedos vizinhos
  - 10 distâncias entre as pontas dos dedos (medidas em "tamanhos de mão")
  -  6 números de orientação: para onde apontam os dedos e para onde a palma está virada

Usar as coordenadas cruas (x, y, z) acerta pouco com uma pessoa que o modelo
nunca viu, porque o modelo aprende a posição e o tamanho da mão de quem gravou.
"""

import itertools
import numpy as np

# Largura / altura da câmera usada nas gravações (640x480).
# O MediaPipe divide x pela largura e y pela altura; multiplicar x (e z, que usa
# a mesma escala de x) por essa proporção deixa as distâncias iguais nos dois eixos.
# Ao vivo, passe a proporção do frame realmente processado (ver extrair).
PROPORCAO_IMAGEM = 640 / 480

NOMES_DEDOS = ["polegar", "indicador", "medio", "anelar", "minimo"]

# Pontos de cada dedo, do pulso (0) até a ponta
DEDOS = [
    [0, 1, 2, 3, 4],
    [0, 5, 6, 7, 8],
    [0, 9, 10, 11, 12],
    [0, 13, 14, 15, 16],
    [0, 17, 18, 19, 20],
]

# Pares de dedos vizinhos (base -> ponta) para medir a abertura entre eles
DEDOS_VIZINHOS = [
    ("polegar_indicador", (1, 4), (5, 8)),
    ("indicador_medio", (5, 8), (9, 12)),
    ("medio_anelar", (9, 12), (13, 16)),
    ("anelar_minimo", (13, 16), (17, 20)),
]

PONTAS = [4, 8, 12, 16, 20]

NOMES = (
    [f"dobra_{nome}_{k}" for nome in NOMES_DEDOS for k in (1, 2, 3)]
    + [f"abertura_{nome}" for nome, _, _ in DEDOS_VIZINHOS]
    + [f"dist_{NOMES_DEDOS[a]}_{NOMES_DEDOS[b]}" for a, b in itertools.combinations(range(5), 2)]
    + ["dedos_x", "dedos_y", "dedos_z", "palma_x", "palma_y", "palma_z"]
)


def _unitario(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _angulo(a, b):
    """Ângulo em graus entre vetores (n, 3)."""
    cos = (_unitario(a) * _unitario(b)).sum(axis=-1)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def mao_esquerda(nome_mediapipe):
    """Converte o nome de mão do MediaPipe ('Left'/'Right') em True = mão esquerda.

    Vale para frames ESPELHADOS (cv2.flip(frame, 1)), como os do coletor e do
    classificador ao vivo. Nesses frames o MediaPipe chama a mão direita do
    usuário de 'Left' e a esquerda de 'Right'. Medido nos vídeos gravados:
    ~99% dos frames do Fernando e ~97% das tomadas do Gabriel com nome certo.
    """
    if nome_mediapipe == "Left":
        return False
    if nome_mediapipe == "Right":
        return True
    raise ValueError(f"Nome de mão desconhecido: {nome_mediapipe!r} (esperado 'Left' ou 'Right')")


def extrair(pontos, esquerda, proporcao=PROPORCAO_IMAGEM):
    """Calcula as características de vários frames de uma vez.

    pontos:    array (n, 21, 3) com x, y, z de cada landmark, como o MediaPipe entrega
               (só frames COM mão; frame sem mão não tem característica)
    esquerda:  array (n,) de bool, True quando o frame é da mão esquerda
    proporcao: largura / altura da imagem de onde vieram os landmarks
               (padrão: 640/480, a resolução de todas as gravações)
    Retorna um array (n, 35), na ordem de NOMES.
    """
    p = np.array(pontos, dtype=float)
    p[..., 0] *= proporcao
    p[..., 2] *= proporcao

    # Pulso na origem e mão esquerda espelhada: vira uma mão direita
    p -= p[:, :1, :]
    p[np.asarray(esquerda, dtype=bool), :, 0] *= -1

    # Tamanho da mão: média das distâncias do pulso às bases dos 4 dedos
    # (a mesma medida que o coletor mostra na tela)
    tamanho = np.linalg.norm(p[:, [5, 9, 13, 17], :2], axis=2).mean(axis=1)
    p /= tamanho[:, None, None]

    dobras = [
        180.0 - _angulo(p[:, a] - p[:, b], p[:, c] - p[:, b])   # 0 = dedo esticado
        for dedo in DEDOS
        for a, b, c in zip(dedo, dedo[1:], dedo[2:])
    ]
    aberturas = [
        _angulo(p[:, a2] - p[:, a1], p[:, b2] - p[:, b1])
        for _, (a1, a2), (b1, b2) in DEDOS_VIZINHOS
    ]
    distancias = [
        np.linalg.norm(p[:, PONTAS[a]] - p[:, PONTAS[b]], axis=1)
        for a, b in itertools.combinations(range(5), 2)
    ]

    # Orientação: 'dedos' aponta do pulso para a base do dedo médio;
    # 'palma' é perpendicular à palma
    dedos = _unitario(p[:, 9])
    lateral = p[:, 5] - p[:, 17]
    lateral = _unitario(lateral - (lateral * dedos).sum(axis=1, keepdims=True) * dedos)
    palma = np.cross(lateral, dedos)

    return np.column_stack(dobras + aberturas + distancias + [dedos, palma])
