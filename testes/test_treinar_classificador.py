"""Cenários da spec treino-classificador (correção da mão por tomada)."""
import contextlib
import io
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import treinar_classificador as t

COLUNAS = (["frame_id"] + [f"lm_{i}_{c}" for i in range(21) for c in ("x", "y", "z", "vis")]
           + ["hand_confidence", "label"])


def gravar_tomada(pasta, arquivo, label, n=4):
    gerador = np.random.default_rng(len(arquivo))
    linhas = []
    for f in range(1, n + 1):
        pontos = gerador.uniform(0.3, 0.7, size=(21, 4))
        pontos[:, 3] = 0.0
        linhas.append([f, *pontos.ravel(), 0.9, label])
    pd.DataFrame(linhas, columns=COLUNAS).to_csv(os.path.join(pasta, arquivo), index=False)


class TestCorrecaoMao(unittest.TestCase):
    def setUp(self):
        self.pasta = tempfile.mkdtemp()
        gravar_tomada(self.pasta, "letra a esquerda.csv", "a")
        gravar_tomada(self.pasta, "letra b esquerda.csv", "b")
        gravar_tomada(self.pasta, "letra c direita_2.csv", "c")

    def ler(self, correcoes):
        saida = io.StringIO()
        with mock.patch.object(t, "CORRECOES_MAO", correcoes), mock.patch.object(t, "CORRECOES_LABEL", {}), \
                contextlib.redirect_stdout(saida):
            dados = t.ler_dataset("Teste", self.pasta)
        lado = dados.groupby("tomada")["esquerda"].agg(lambda s: set(s))
        return {tomada.split("/", 1)[1]: valores for tomada, valores in lado.items()}, saida.getvalue()

    def test_tomada_da_tabela_usa_mao_corrigida(self):
        lado, saida = self.ler({("Teste", "letra a esquerda.csv"): "direita",
                                ("Teste", "letra c direita_2.csv"): "esquerda"})
        self.assertEqual(lado["letra a esquerda.csv"], {False})
        self.assertEqual(lado["letra c direita_2.csv"], {True})
        self.assertIn("2 tomadas com a mão corrigida", saida)

    def test_tomada_fora_da_tabela_usa_nome(self):
        lado, _ = self.ler({("Teste", "letra a esquerda.csv"): "direita"})
        self.assertEqual(lado["letra b esquerda.csv"], {True})
        self.assertEqual(lado["letra c direita_2.csv"], {False})

    def test_outro_dataset_nao_e_afetado(self):
        lado, saida = self.ler({("Outra pessoa", "letra a esquerda.csv"): "direita"})
        self.assertEqual(lado["letra a esquerda.csv"], {True})
        self.assertIn("0 tomadas com a mão corrigida", saida)
        self.assertNotIn("não existe", saida)

    def test_entrada_sem_arquivo_avisa_e_continua(self):
        lado, saida = self.ler({("Teste", "letra z esquerda.csv"): "direita"})
        self.assertIn("'letra z esquerda.csv', que não existe", saida)
        self.assertEqual(len(lado), 3)

    def test_arquivos_nao_sao_alterados(self):
        antes = {a: os.path.getmtime(os.path.join(self.pasta, a)) for a in os.listdir(self.pasta)}
        self.ler({("Teste", "letra a esquerda.csv"): "direita"})
        depois = {a: os.path.getmtime(os.path.join(self.pasta, a)) for a in os.listdir(self.pasta)}
        self.assertEqual(antes, depois)


class TestTabelaReal(unittest.TestCase):
    def test_maos_validas(self):
        self.assertTrue(set(t.CORRECOES_MAO.values()) <= {"direita", "esquerda"})

    def test_correcao_inverte_o_nome(self):
        # Só faz sentido corrigir para a mão OPOSTA à do nome do arquivo
        for (_, arquivo), mao in t.CORRECOES_MAO.items():
            self.assertNotIn(mao[:-1], arquivo, arquivo)


if __name__ == "__main__":
    unittest.main()
