"""Cenários da spec caracteristicas-mao."""
import unittest

import numpy as np

import caracteristicas_mao as cm


def mao(semente=0, n=3):
    """n frames de 21 landmarks (x, y, z normalizados) com formatos diferentes."""
    gerador = np.random.default_rng(semente)
    pontos = gerador.uniform(0.3, 0.7, size=(n, 21, 3))
    pontos[..., 2] = gerador.uniform(-0.05, 0.05, size=(n, 21))
    return pontos


class TestCalculoUnico(unittest.TestCase):
    def test_nomes_na_ordem_das_colunas(self):
        X = cm.extrair(mao(), np.zeros(3, bool))
        self.assertEqual(X.shape, (3, len(cm.NOMES)))
        self.assertEqual(len(cm.NOMES), 35)
        self.assertEqual(len(set(cm.NOMES)), 35)

    def test_mesmos_landmarks_mesmos_valores(self):
        p = mao()
        np.testing.assert_array_equal(cm.extrair(p, np.zeros(3, bool)), cm.extrair(p.copy(), np.zeros(3, bool)))

    def test_nao_altera_entrada(self):
        p = mao()
        copia = p.copy()
        cm.extrair(p, np.ones(3, bool))
        np.testing.assert_array_equal(p, copia)


class TestInvariancia(unittest.TestCase):
    def setUp(self):
        self.p = mao()
        self.direita = np.zeros(3, bool)
        self.base = cm.extrair(self.p, self.direita)

    def test_mao_deslocada(self):
        deslocada = self.p + np.array([0.12, -0.07, 0.0])
        np.testing.assert_allclose(cm.extrair(deslocada, self.direita), self.base, atol=1e-9)

    def test_mao_maior(self):
        pulso = self.p[:, :1, :]
        maior = pulso + (self.p - pulso) * 1.8
        np.testing.assert_allclose(cm.extrair(maior, self.direita), self.base, atol=1e-9)

    def test_mao_esquerda_espelhada(self):
        espelhada = self.p.copy()
        espelhada[..., 0] = 1.0 - espelhada[..., 0]
        np.testing.assert_allclose(cm.extrair(espelhada, np.ones(3, bool)), self.base, atol=1e-9)

    def test_espelhada_sem_avisar_muda(self):
        espelhada = self.p.copy()
        espelhada[..., 0] = 1.0 - espelhada[..., 0]
        self.assertFalse(np.allclose(cm.extrair(espelhada, self.direita), self.base))


class TestProporcao(unittest.TestCase):
    def test_padrao_e_640_por_480(self):
        p = mao()
        lado = np.zeros(3, bool)
        np.testing.assert_array_equal(cm.extrair(p, lado), cm.extrair(p, lado, proporcao=640 / 480))

    def test_mesma_mao_em_1280x720(self):
        # Mão em pixels numa imagem 640x480 e a mesma mão (1,5x maior) numa 1280x720
        px = mao() * np.array([640, 480, 640])
        em_640 = px / np.array([640, 480, 640])
        em_1280 = px * 1.5 / np.array([1280, 720, 1280])
        lado = np.zeros(3, bool)
        np.testing.assert_allclose(cm.extrair(em_1280, lado, proporcao=1280 / 720),
                                   cm.extrair(em_640, lado, proporcao=640 / 480), atol=1e-9)

    def test_proporcao_errada_muda_resultado(self):
        px = mao() * np.array([640, 480, 640])
        em_1280 = px * 1.5 / np.array([1280, 720, 1280])
        lado = np.zeros(3, bool)
        self.assertFalse(np.allclose(cm.extrair(em_1280, lado),
                                     cm.extrair(em_1280, lado, proporcao=1280 / 720)))


class TestConversaoMao(unittest.TestCase):
    def test_left_e_mao_direita(self):
        self.assertFalse(cm.mao_esquerda("Left"))

    def test_right_e_mao_esquerda(self):
        self.assertTrue(cm.mao_esquerda("Right"))

    def test_nome_desconhecido(self):
        for nome in ("left", "", None, "Direita"):
            with self.assertRaises(ValueError):
                cm.mao_esquerda(nome)


if __name__ == "__main__":
    unittest.main()
