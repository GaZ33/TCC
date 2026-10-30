"""Testes do aumento de dados."""
import unittest

import numpy as np

import aumento_dados as a
import caracteristicas_mao as cm

N_ANGULOS = 19
N_DISTANCIAS = 10


def mao(n=4, semente=0):
    gerador = np.random.default_rng(semente)
    pontos = gerador.uniform(0.3, 0.7, size=(n, 21, 3))
    pontos[..., 2] = gerador.uniform(-0.05, 0.05, size=(n, 21))
    return pontos


class TestAumento(unittest.TestCase):
    def setUp(self):
        self.p = mao()
        self.lado = np.array([False, True, False, True])

    def gerar(self, **opcoes):
        return a.aumentar(self.p, self.lado, 3, np.random.default_rng(1), **opcoes)

    def test_formato_e_lado(self):
        pontos, lado = self.gerar()
        self.assertEqual(pontos.shape, (12, 21, 3))
        np.testing.assert_array_equal(lado, np.repeat(self.lado, 3))

    def test_sem_variacao_devolve_copias_iguais(self):
        pontos, _ = self.gerar(rotacao=0, proporcao_dedos=0, ruido=0)
        np.testing.assert_allclose(pontos, np.repeat(self.p, 3, axis=0))

    def test_rotacao_mantem_os_angulos_e_muda_a_orientacao(self):
        pontos, lado = self.gerar(proporcao_dedos=0, ruido=0)
        original = cm.extrair(np.repeat(self.p, 3, axis=0), lado)
        aumentado = cm.extrair(pontos, lado)
        # Ângulos em 3D não dependem da rotação; a orientação (6 últimas) muda.
        # (As distâncias mudam um pouco: o tamanho da mão que as divide é medido só em x e y.)
        np.testing.assert_allclose(aumentado[:, :N_ANGULOS], original[:, :N_ANGULOS], atol=1e-6)
        self.assertFalse(np.allclose(aumentado[:, -6:], original[:, -6:]))

    def test_proporcao_dos_dedos_muda_distancias_e_nao_os_angulos(self):
        pontos, lado = self.gerar(rotacao=0, ruido=0)
        original = cm.extrair(np.repeat(self.p, 3, axis=0), lado)
        aumentado = cm.extrair(pontos, lado)
        dobras = slice(0, 15)   # o ângulo de cada articulação não muda com o comprimento do dedo
        np.testing.assert_allclose(aumentado[:, dobras], original[:, dobras], atol=1e-6)
        self.assertFalse(np.allclose(aumentado[:, N_ANGULOS:N_ANGULOS + N_DISTANCIAS],
                                     original[:, N_ANGULOS:N_ANGULOS + N_DISTANCIAS]))

    def test_pulso_e_bases_dos_dedos_nao_se_movem_sem_rotacao_e_ruido(self):
        pontos, _ = self.gerar(rotacao=0, ruido=0)
        fixos = [0, 1, 5, 9, 13, 17]
        np.testing.assert_allclose(pontos[:, fixos], np.repeat(self.p, 3, axis=0)[:, fixos])

    def test_reprodutivel_com_a_mesma_semente(self):
        np.testing.assert_array_equal(self.gerar()[0], self.gerar()[0])


if __name__ == "__main__":
    unittest.main()
