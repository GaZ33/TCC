"""Testes do treino: leitura dos CSVs, avaliação "pessoa de fora" e arquivos de resultado."""
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

# Uma "mão" diferente por letra: frames da mesma letra variam só um pouco
MAOS = {letra: np.random.default_rng(i).uniform(0.3, 0.7, size=(21, 3)) for i, letra in enumerate("abc")}


def gravar_tomada(pasta, arquivo, label, n=6, sem_mao=0, semente=0):
    gerador = np.random.default_rng(semente)
    linhas = []
    for f in range(1, n + 1):
        if f <= sem_mao:
            linhas.append([f] + [0.0] * 85 + [label])
            continue
        pontos = np.zeros((21, 4))
        pontos[:, :3] = MAOS[label] + gerador.normal(0, 0.003, size=(21, 3))
        linhas.append([f, *pontos.ravel(), 0.9, label])
    pd.DataFrame(linhas, columns=COLUNAS).to_csv(os.path.join(pasta, arquivo), index=False)


def ler(pasta, nome="Teste"):
    saida = io.StringIO()
    with contextlib.redirect_stdout(saida):
        dados, info = t.ler_dataset(nome, pasta)
    return dados, info, saida.getvalue()


class TestLeitura(unittest.TestCase):
    def setUp(self):
        self.pasta = tempfile.mkdtemp()

    def test_mao_vem_do_nome_do_arquivo(self):
        for arquivo in ("letra a direita.csv", "letra b esquerdo.csv", "letra c_esquerda_2.csv"):
            gravar_tomada(self.pasta, arquivo, arquivo[6])
        dados, info, _ = ler(self.pasta)
        lado = dados.groupby("tomada")["esquerda"].agg(set)
        self.assertEqual(lado["Teste/letra a direita.csv"], {False})
        self.assertEqual(lado["Teste/letra b esquerdo.csv"], {True})
        self.assertEqual(lado["Teste/letra c_esquerda_2.csv"], {True})
        self.assertEqual(info["tomadas"], 3)

    def test_frames_sem_mao_sao_descartados(self):
        gravar_tomada(self.pasta, "letra a direita.csv", "a", n=6, sem_mao=2)
        dados, info, _ = ler(self.pasta)
        self.assertEqual(len(dados), 4)
        self.assertEqual(info["frames_sem_mao"], 2)

    def test_variante_no_nome_usa_a_primeira_letra(self):
        gravar_tomada(self.pasta, "letra a2 direito.csv", "a")   # como "letra 1a"/"letra 1b" do Giovanni
        dados, _, saida = ler(self.pasta)
        self.assertEqual(set(dados["label"]), {"a"})
        self.assertNotIn("[AVISO]", saida)

    def test_label_diferente_do_nome_e_ignorado(self):
        gravar_tomada(self.pasta, "letra b direita.csv", "a")
        gravar_tomada(self.pasta, "letra a direita.csv", "a")
        dados, _, saida = ler(self.pasta)
        self.assertIn("'letra b direita.csv' tem label ['a'], mas o nome diz 'b'", saida)
        self.assertEqual(set(dados["tomada"]), {"Teste/letra a direita.csv"})

    def test_arquivo_fora_do_padrao_e_ignorado(self):
        gravar_tomada(self.pasta, "letra a direita.csv", "a")
        gravar_tomada(self.pasta, "gravacao.csv", "a")
        dados, _, saida = ler(self.pasta)
        self.assertIn("'gravacao.csv' fora do padrão", saida)
        self.assertEqual(set(dados["tomada"]), {"Teste/letra a direita.csv"})

    def test_pasta_vazia(self):
        with self.assertRaises(SystemExit):
            ler(self.pasta)


class TestAvaliacao(unittest.TestCase):
    def setUp(self):
        self.datasets = {}
        for p, pessoa in enumerate(("P1", "P2", "P3")):
            pasta = tempfile.mkdtemp()
            for letra in "abc":
                gravar_tomada(pasta, f"letra {letra} direita.csv", letra, semente=10 * p + ord(letra))
            self.datasets[pessoa] = ler(pasta, pessoa)[0]
        self.patch = mock.patch.object(t, "N_ARVORES", 10)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()

    def test_cada_pessoa_e_testada_fora_do_treino(self):
        previsoes, fora = t.avaliar_pessoa_de_fora(self.datasets)
        self.assertEqual(set(previsoes["pessoa"]), {"P1", "P2", "P3"})
        self.assertEqual(len(previsoes), sum(len(d) for d in self.datasets.values()))
        self.assertEqual(fora, {"P1": [], "P2": [], "P3": []})
        # Letras bem separadas: o modelo acerta mesmo com a pessoa de fora
        self.assertGreater((previsoes["real"] == previsoes["previsto"]).mean(), 0.9)

    def test_classe_que_so_a_pessoa_tem_fica_fora_da_conta(self):
        pasta = tempfile.mkdtemp()
        gravar_tomada(pasta, "letra a direita.csv", "a")
        gravar_tomada(pasta, "letra c direita.csv", "c")
        self.datasets["P3"] = ler(pasta, "P3")[0]
        self.datasets["P1"] = self.datasets["P1"][self.datasets["P1"]["label"] != "c"]
        self.datasets["P2"] = self.datasets["P2"][self.datasets["P2"]["label"] != "c"]
        previsoes, fora = t.avaliar_pessoa_de_fora(self.datasets)
        self.assertEqual(fora["P3"], ["c"])
        self.assertNotIn("c", set(previsoes.loc[previsoes["pessoa"] == "P3", "real"]))

    def test_voto_por_tomada(self):
        previsoes = pd.DataFrame({"pessoa": "P1", "tomada": ["t1"] * 5 + ["t2"] * 3,
                                  "real": ["a"] * 5 + ["b"] * 3,
                                  "previsto": list("aabaa") + list("ccb")})
        tomadas = t.votar_por_tomada(previsoes)
        self.assertEqual(tomadas.loc["t1", "previsto"], "a")
        self.assertEqual(tomadas.loc["t2", "previsto"], "c")

    def test_resumo_tem_media_das_pessoas(self):
        previsoes = pd.DataFrame({"pessoa": ["P1"] * 4 + ["P2"] * 2, "tomada": ["t1"] * 4 + ["t2"] * 2,
                                  "real": ["a"] * 6, "previsto": list("aaab") + list("bb")})
        resumo = t.resumir_por_pessoa(previsoes)
        self.assertAlmostEqual(resumo.loc["P1", "acerto_frame"], 0.75)
        self.assertAlmostEqual(resumo.loc["P2", "acerto_frame"], 0.0)
        self.assertAlmostEqual(resumo.loc["Média", "acerto_frame"], 0.375)   # cada pessoa pesa igual
        self.assertEqual(resumo.loc["Média", "frames"], 6)


class TestResultados(unittest.TestCase):
    def test_arquivos_gerados(self):
        previsoes = pd.DataFrame({"pessoa": ["P1"] * 4 + ["P2"] * 4, "tomada": ["t1"] * 4 + ["t2"] * 4,
                                  "real": list("aabb") * 2, "previsto": list("abbb") + list("aaba")})
        resumo = t.resumir_por_pessoa(previsoes)
        info = {"P1": {"tomadas": 1, "frames": 4, "frames_sem_mao": 0},
                "P2": {"tomadas": 1, "frames": 4, "frames_sem_mao": 2}}
        pasta = tempfile.mkdtemp()
        caminhos = t.salvar_resultados(pasta, info, resumo, previsoes, {"P1": [], "P2": []}, 0.99,
                                       os.path.join(t.SCRIPT_DIR, "modelos", "m.joblib"))
        for caminho in caminhos.values():
            self.assertTrue(os.path.getsize(caminho) > 0, caminho)
        with open(caminhos["relatorio.md"], encoding="utf-8") as arquivo:
            relatorio = arquivo.read()
        self.assertIn("| **Média** |", relatorio)
        self.assertIn("99,0%", relatorio)
        self.assertIn("modelos/m.joblib", relatorio)
        matriz = pd.read_csv(caminhos["matriz_confusao.csv"], index_col=0)
        np.testing.assert_allclose(matriz.sum(axis=1), 1.0)   # cada linha soma 100%


if __name__ == "__main__":
    unittest.main()
