"""Cenários da spec classificacao-ao-vivo (lógica, sem câmera)."""
import contextlib
import io
import os
import tempfile
import unittest

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier

import caracteristicas_mao
import classificar_ao_vivo as c


class TestVotacao(unittest.TestCase):
    def votar(self, letras):
        v = c.VotacaoLetra()
        for letra in letras:
            v.com_mao(letra)
        return v

    def test_comeca_sem_mao(self):
        v = c.VotacaoLetra()
        self.assertTrue(v.mao_ausente)
        self.assertEqual(v.resultado(), (None, 0))
        v.sem_mao()
        self.assertTrue(v.mao_ausente)

    def test_previsoes_consistentes(self):
        v = self.votar(list("aaaabaaaca"))
        self.assertEqual(v.resultado(), ("a", 8))

    def test_previsoes_divididas(self):
        v = self.votar(list("mmmmnnnwww"))
        self.assertEqual(v.resultado(), (None, 4))

    def test_poucos_frames(self):
        v = self.votar(list("bbb"))
        self.assertEqual(v.resultado(), (None, 3))

    def test_so_os_ultimos_10_votam(self):
        v = self.votar(list("aaaaaaaaaa") + list("bbbbbbb"))
        self.assertEqual(v.resultado(), ("b", 7))

    def test_falha_rapida_de_deteccao(self):
        v = self.votar(list("bbbbbbbbbb"))
        for _ in range(4):
            v.sem_mao()
        self.assertFalse(v.mao_ausente)
        self.assertEqual(v.resultado(), ("b", 10))

    def test_mao_retirada_da_tela(self):
        v = self.votar(list("bbbbbbbbbb"))
        for _ in range(5):
            v.sem_mao()
        self.assertTrue(v.mao_ausente)
        self.assertEqual(v.resultado(), (None, 0))
        v.com_mao("c")
        self.assertFalse(v.mao_ausente)
        self.assertEqual(v.resultado(), (None, 1))

    def test_falha_nao_acumula_entre_aparicoes(self):
        v = self.votar(list("bbbbbbbbbb"))
        for _ in range(3):
            v.sem_mao()
        v.com_mao("b")
        for _ in range(3):
            v.sem_mao()
        self.assertEqual(v.resultado()[0], "b")


class TestEscolhaMao(unittest.TestCase):
    def test_ordem_do_tab(self):
        self.assertEqual(c.proximo_modo("auto"), "direita")
        self.assertEqual(c.proximo_modo("direita"), "esquerda")
        self.assertEqual(c.proximo_modo("esquerda"), "auto")

    def test_modo_automatico(self):
        self.assertFalse(c.mao_em_uso("auto", "Left"))    # mão direita do usuário no frame espelhado
        self.assertTrue(c.mao_em_uso("auto", "Right"))

    def test_mao_fixada(self):
        for nome in ("Left", "Right"):
            self.assertTrue(c.mao_em_uso("esquerda", nome))
            self.assertFalse(c.mao_em_uso("direita", nome))


class ModeloFalso:
    classes_ = np.array(["h", "k", "v", "x", "y"])

    def predict_proba(self, X):
        self.X = X
        return np.array([[0.40, 0.45, 0.05, 0.03, 0.02]])


class TestMaisProvaveis(unittest.TestCase):
    def test_tres_mais_provaveis_em_ordem(self):
        modelo = ModeloFalso()
        pontos = np.random.default_rng(0).uniform(0.3, 0.7, size=(21, 3))
        top = c.classificar_frame({"modelo": modelo}, pontos, False, 640 / 480)
        self.assertEqual([letra for letra, _ in top], ["k", "h", "v"])
        np.testing.assert_allclose([p for _, p in top], [0.45, 0.40, 0.05])
        self.assertEqual(modelo.X.shape, (1, len(caracteristicas_mao.NOMES)))

    def test_mesmas_caracteristicas_do_treino(self):
        modelo = ModeloFalso()
        pontos = np.random.default_rng(1).uniform(0.3, 0.7, size=(21, 3))
        c.classificar_frame({"modelo": modelo}, pontos, True, 1280 / 720)
        esperado = caracteristicas_mao.extrair(pontos[None], [True], 1280 / 720)
        np.testing.assert_array_equal(modelo.X, esperado)


class TestValidacaoModelo(unittest.TestCase):
    def salvar(self, **campos):
        rf = RandomForestClassifier(n_estimators=3, n_jobs=-1, random_state=0)
        rf.fit(np.random.default_rng(0).normal(size=(6, 35)), list("aabbcc"))
        pacote = {"modelo": rf, "classes": ["a", "b", "c"], "caracteristicas": caracteristicas_mao.NOMES,
                  "versao_sklearn": sklearn.__version__, **campos}
        caminho = os.path.join(tempfile.mkdtemp(), "modelo.joblib")
        joblib.dump(pacote, caminho)
        return caminho

    def carregar(self, caminho):
        saida = io.StringIO()
        with contextlib.redirect_stdout(saida):
            pacote = c.carregar_modelo(caminho)
        return pacote, saida.getvalue()

    def test_modelo_compativel(self):
        pacote, saida = self.carregar(self.salvar())
        self.assertEqual(pacote["modelo"].n_jobs, 1)
        self.assertEqual(saida, "")

    def test_modelo_ainda_nao_treinado(self):
        with self.assertRaises(c.ErroModelo) as erro:
            c.carregar_modelo(os.path.join(tempfile.mkdtemp(), "nao_existe.joblib"))
        self.assertIn("treinar_classificador.py", str(erro.exception))

    def test_outras_caracteristicas(self):
        caminho = self.salvar(caracteristicas=caracteristicas_mao.NOMES[:-1])
        with self.assertRaises(c.ErroModelo) as erro:
            c.carregar_modelo(caminho)
        self.assertIn("Treine de novo", str(erro.exception))

    def test_arquivo_de_outro_programa(self):
        caminho = os.path.join(tempfile.mkdtemp(), "outro.joblib")
        joblib.dump([1, 2, 3], caminho)
        with self.assertRaises(c.ErroModelo):
            c.carregar_modelo(caminho)

    def test_versao_diferente_do_sklearn(self):
        pacote, saida = self.carregar(self.salvar(versao_sklearn="0.0.1"))
        self.assertIn("[AVISO]", saida)
        self.assertIn("0.0.1", saida)
        self.assertIn("modelo", pacote)

    @unittest.skipUnless(os.path.exists(c.CAMINHO_MODELO), "modelo ainda não treinado")
    def test_modelo_salvo_pelo_treino(self):
        pacote, _ = self.carregar(c.CAMINHO_MODELO)
        self.assertTrue(all(isinstance(classe, str) for classe in pacote["classes"]))
        self.assertEqual(pacote["classes"], [str(x) for x in pacote["modelo"].classes_])


if __name__ == "__main__":
    unittest.main()
