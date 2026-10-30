"""
Aumento de dados: gera frames artificiais a partir de frames reais.

Cada cópia artificial é o mesmo frame com pequenas variações que poderiam acontecer
com outra pessoa ou em outra gravação:
  - rotação 3D da mão em torno do pulso (até ROTACAO_MAX_GRAUS): outra orientação da palma;
  - proporção dos dedos (cada dedo, independente, entre -PROPORCAO_DEDOS e +PROPORCAO_DEDOS):
    mãos com dedos mais longos ou mais curtos;
  - ruído nos pontos (desvio de RUIDO tamanhos de mão): a tremedeira do detector.

Os pontos artificiais saem no mesmo formato do MediaPipe (x, y, z normalizados pela imagem),
então passam pela mesma função de características do treino. Só o TREINO recebe dados
artificiais; o teste usa sempre frames reais.
"""

import numpy as np

import caracteristicas_mao

ROTACAO_MAX_GRAUS = 15.0
PROPORCAO_DEDOS = 0.10
RUIDO = 0.01


def _rotacoes_aleatorias(n, max_graus, gerador):
    """n matrizes de rotação 3D (eixo aleatório, ângulo uniforme em ±max_graus), fórmula de Rodrigues."""
    eixo = gerador.normal(size=(n, 3))
    eixo /= np.linalg.norm(eixo, axis=1, keepdims=True)
    angulo = np.radians(gerador.uniform(-max_graus, max_graus, size=n))
    x, y, z = eixo.T
    zero = np.zeros(n)
    K = np.stack([np.stack([zero, -z, y], 1), np.stack([z, zero, -x], 1), np.stack([-y, x, zero], 1)], 1)
    seno, cosseno = np.sin(angulo)[:, None, None], np.cos(angulo)[:, None, None]
    return np.eye(3) + seno * K + (1 - cosseno) * (K @ K)


def aumentar(pontos, esquerda, n_copias, gerador, rotacao=ROTACAO_MAX_GRAUS, proporcao_dedos=PROPORCAO_DEDOS,
             ruido=RUIDO, proporcao_imagem=caracteristicas_mao.PROPORCAO_IMAGEM):
    """Gera n_copias frames artificiais por frame real.

    pontos:    (n, 21, 3) landmarks como o MediaPipe entrega (só frames com mão)
    esquerda:  (n,) bool, True quando o frame é da mão esquerda
    rotacao, proporcao_dedos, ruido: intensidade de cada variação (0 desliga)
    Retorna (pontos_artificiais (n * n_copias, 21, 3), esquerda_artificial (n * n_copias,)).
    """
    p = np.repeat(np.asarray(pontos, dtype=float), n_copias, axis=0)
    lado = np.repeat(np.asarray(esquerda, dtype=bool), n_copias)
    n = len(p)

    # Mesma escala nos 3 eixos e pulso na origem (as variações são em torno do pulso)
    p[..., 0] *= proporcao_imagem
    p[..., 2] *= proporcao_imagem
    pulso = p[:, :1, :].copy()
    p -= pulso

    if proporcao_dedos:
        for dedo in caracteristicas_mao.DEDOS:
            fator = gerador.uniform(1 - proporcao_dedos, 1 + proporcao_dedos, size=(n, 1))
            novo = p.copy()
            # A base do dedo fica no lugar; cada segmento seguinte cresce ou encolhe pelo fator
            for anterior, atual in zip(dedo[1:], dedo[2:]):
                novo[:, atual] = novo[:, anterior] + fator * (p[:, atual] - p[:, anterior])
            p = novo

    if rotacao:
        p = np.einsum("nij,nkj->nki", _rotacoes_aleatorias(n, rotacao, gerador), p)

    if ruido:
        tamanho = np.linalg.norm(p[:, [5, 9, 13, 17], :], axis=2).mean(axis=1)
        p += gerador.normal(0.0, 1.0, size=p.shape) * (ruido * tamanho)[:, None, None]

    p += pulso
    p[..., 0] /= proporcao_imagem
    p[..., 2] /= proporcao_imagem
    return p, lado
