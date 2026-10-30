"""Testes do avaliador de vídeos (sem MediaPipe: pontos e modelo falsos)."""
import unittest

import numpy as np

import avaliar_videos as av


class ModeloFalso:
    """Imita classificador.ModeloLibras: devolve sempre as mesmas probabilidades para (a, b, c)."""
    classes = ("a", "b", "c")
    proporcao_treino = 640 / 480

    def __init__(self, prob_a):
        resto = (1 - prob_a) / 2
        self.probabilidades = [prob_a, resto, resto]

    def inferir(self, pontos, esquerda, proporcao):
        n = len(pontos)
        return np.tile(self.probabilidades, (n, 1)), np.ones(n, dtype=bool)


def mao(n):
    gerador = np.random.default_rng(0)
    return gerador.uniform(0.3, 0.7, size=(n, 21, 3))


class TestCadeia(unittest.TestCase):
    def test_primeira_letra_no_primeiro_envio(self):
        mostradas = av.simular_cadeia(ModeloFalso(0.8), mao(90), np.full(90, 0.9), ["Left"] * 90, 640 / 480)
        self.assertEqual(mostradas[0], (av.STEP - 1, "a"))
        self.assertEqual(len(mostradas), 90 // av.STEP)

    def test_confianca_baixa_nao_mostra(self):
        mostradas = av.simular_cadeia(ModeloFalso(0.45), mao(90), np.full(90, 0.9), ["Left"] * 90, 640 / 480)
        self.assertEqual(mostradas, [])

    def test_sem_mao_nao_mostra(self):
        mostradas = av.simular_cadeia(ModeloFalso(0.9), np.zeros((90, 21, 3)), np.zeros(90), [None] * 90, 640 / 480)
        self.assertEqual(mostradas, [])

    def test_mao_so_na_metade_da_janela(self):
        # 15 de 30 quadros com mão: o blank "-" leva metade da probabilidade e a letra não passa de 50%
        confianca = np.tile(np.r_[np.full(15, 0.9), np.zeros(15)], 3)
        nomes = ["Left" if c else None for c in confianca]
        mostradas = av.simular_cadeia(ModeloFalso(0.9), mao(90), confianca, nomes, 640 / 480)
        self.assertEqual(mostradas, [])


class TestEspelhamento(unittest.TestCase):
    def test_video_do_celular_sem_espelho(self):
        # Mão direita no quadro espelhado pelo avaliador: o MediaPipe diz "Left" (convenção do sistema)
        pontos = mao(4)
        corrigidos, nomes, espelhado = av.corrigir_espelhamento(pontos, ["Left", "Left", "Right", None])
        self.assertFalse(espelhado)
        np.testing.assert_array_equal(corrigidos, pontos)

    def test_video_ja_espelhado(self):
        pontos = mao(4)
        corrigidos, nomes, espelhado = av.corrigir_espelhamento(pontos, ["Right", "Right", "Left", None])
        self.assertTrue(espelhado)
        self.assertEqual(nomes, ["Left", "Left", "Right", None])
        np.testing.assert_allclose(corrigidos[:3, :, 0], 1 - pontos[:3, :, 0])
        np.testing.assert_array_equal(corrigidos[3], pontos[3])   # quadro sem mão fica como estava

    def test_sem_mao_nenhuma(self):
        _, _, espelhado = av.corrigir_espelhamento(np.zeros((3, 21, 3)), [None] * 3)
        self.assertFalse(espelhado)


if __name__ == "__main__":
    unittest.main()
