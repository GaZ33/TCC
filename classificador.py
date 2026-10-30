#!/usr/bin/env python3
"""
classificador_zmq.py: classificador de letras do alfabeto manual em tempo real.

Posição na arquitetura (todos os elos são PUSH -> PULL; quem recebe faz ``bind``)::

    Emissor --(5555, pickle)--> CLASSIFICADOR --(5556, JSON)--> Filtro --(5557, JSON)--> Exibidor

Este processo:

1. recebe do emissor a janela deslizante de frames (``tcp://127.0.0.1:5555``, PULL);
2. descarta janelas atrasadas e frames já classificados (as janelas se sobrepõem);
3. extrai as 35 características invariantes de cada frame novo com mão
   (``caracteristicas_mao.extrair``) e calcula ``predict_proba`` do Random Forest;
4. envia ao filtro (``tcp://127.0.0.1:5556``, PUSH) um dicionário JSON de
   probabilidades por classe, em porcentagem, com ``"-"`` como classe blank::

       {"a": 85.2, "b": 1.5, ..., "-": 5.0}

Contrato do pacote recebido (``send_pyobj`` do emissor)
-------------------------------------------------------
Campos obrigatórios: ``colunas`` (nomes) e ``amostras`` (uma linha por frame, na ordem
de ``colunas``: ``frame_id``, ``lm_{0..20}_{x,y,z,vis}``, ``hand_confidence``).
Campos opcionais: ``sequencia`` (detecta reinício do emissor), ``maos`` (lista paralela
a ``amostras`` com ``"Left"``/``"Right"``/``None``, nomes do MediaPipe) e ``proporcao``
(largura/altura da câmera). Sem ``maos`` usa-se ``--mao-padrao``; sem ``proporcao``,
a do treino (640/480). Os dois valores alteram as características, então o ideal é o
emissor enviá-los (ver o resumo da entrega).

Significado do blank ``"-"``
----------------------------
O modelo foi treinado só com frames COM mão, então ele não conhece o blank. Aqui o blank é
a fração de frames sem mão (``hand_confidence <= 0``) ou com landmarks inutilizáveis.

Modos de saída (``--modo-saida``)
---------------------------------
``janela`` (padrão): uma mensagem por janela recebida, com a média das probabilidades
dos frames novos. ``frame``: uma mensagem por frame novo. O filtro conta blanks por
MENSAGEM (``BLANKS_PARA_LIMPAR``), então 10 blanks são 10 janelas (5 a 15 s) no modo
``janela`` e 10 frames (0,33 s a 30 fps) no modo ``frame``.

Segurança
---------
* O pacote do emissor é pickle. Ele é lido por um ``Unpickler`` que proíbe qualquer
  objeto global (o pacote só tem dict, list, str, int e float), então um pacote
  malicioso não executa código. Mesmo assim, os sockets ficam em ``127.0.0.1``.
* O modelo ``.joblib`` também é pickle: carregue só arquivos seus. ``--modelo-sha256``
  confere a integridade do arquivo antes de carregá-lo.

Uso::

    python classificador_zmq.py [--modelo CAMINHO] [--modo-saida janela|frame]
"""

from __future__ import annotations

import argparse
import errno
import functools
import hashlib
import io
import json
import logging
import os
import pickle
import signal
import sys
import time
from dataclasses import dataclass
from types import FrameType
from typing import Any, Final, Optional, Sequence

import joblib
import numpy as np
import sklearn
import zmq
from numpy.typing import NDArray

import caracteristicas_mao

# =====================================================================
# CONFIGURAÇÕES
# =====================================================================
SCRIPT_DIR: Final[str] = os.path.dirname(os.path.abspath(__file__))
CAMINHO_MODELO_PADRAO: Final[str] = os.path.join(SCRIPT_DIR, "modelos", "random_forest_letras.joblib")

ENDERECO_ENTRADA_PADRAO: Final[str] = "tcp://127.0.0.1:5555"   # bind (PULL): o emissor conecta aqui
ENDERECO_SAIDA_PADRAO: Final[str] = "tcp://127.0.0.1:5556"     # connect (PUSH): o filtro faz bind aqui

ROTULO_BLANK: Final[str] = "-"
# Rótulos que o filtro aceita; qualquer outro é descartado por ele em silêncio.
ROTULOS_DO_FILTRO: Final[frozenset[str]] = frozenset("abcdefghijklmnopqrstuvwxyz0123456789")

TIMEOUT_POLL_MS: Final[int] = 500                    # também é o tempo máximo para reagir a Ctrl+C
TAMANHO_MAXIMO_PACOTE_BYTES: Final[int] = 8 * 1024 * 1024
LIMITE_LINHAS_POR_PACOTE: Final[int] = 2048          # uma janela normal tem 120 linhas
FRAMES_MAX_POR_PACOTE: Final[int] = 60               # só os frames novos mais recentes são classificados
HWM_ENTRADA: Final[int] = 8                          # fila curta: janela velha não interessa
HWM_SAIDA: Final[int] = 200
INTERVALO_RESUMO_S: Final[float] = 10.0
SEM_PACOTES_AVISO_S: Final[float] = 10.0
LIMITE_LATENCIA_MS: Final[float] = 250.0             # o emissor reenvia a cada 0,5 s no mínimo

N_LANDMARKS: Final[int] = 21
COLUNA_FRAME: Final[str] = "frame_id"
COLUNA_CONFIANCA: Final[str] = "hand_confidence"
MODOS_SAIDA: Final[tuple[str, ...]] = ("janela", "frame")

logger = logging.getLogger("classificador_zmq")


# =====================================================================
# EXCEÇÕES
# =====================================================================
class ErroModelo(Exception):
    """O arquivo do modelo não existe, está adulterado ou não serve para este classificador."""


class PacoteInvalido(ValueError):
    """O pacote recebido está corrompido ou fora do contrato; ele é descartado e o serviço segue."""


# =====================================================================
# MODELO
# =====================================================================
def calcular_sha256(caminho: str) -> str:
    """Calcula o SHA-256 de um arquivo lendo em blocos de 1 MiB (memória constante).

    Args:
        caminho: Caminho do arquivo.

    Returns:
        Resumo em hexadecimal minúsculo.
    """
    resumo = hashlib.sha256()
    with open(caminho, "rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1 << 20), b""):
            resumo.update(bloco)
    return resumo.hexdigest()


class ModeloLibras:
    """Encapsula o Random Forest e a extração de características de um lote de frames."""

    def __init__(self, estimador: Any, classes: tuple[str, ...], proporcao_treino: float) -> None:
        """Cria o invólucro.

        Args:
            estimador: Estimador do scikit-learn com ``predict_proba`` e ``classes_``.
            classes: Rótulos na ordem das colunas de ``predict_proba``.
            proporcao_treino: Largura/altura da imagem usada no treino.
        """
        self._estimador = estimador
        self.classes: tuple[str, ...] = classes
        self.proporcao_treino: float = proporcao_treino

    def inferir(
        self,
        pontos: NDArray[np.float64],
        esquerda: NDArray[np.bool_],
        proporcao: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        """Calcula as probabilidades por classe de um lote de frames COM mão.

        Args:
            pontos: Landmarks ``(n, 21, 3)`` como o MediaPipe entrega (x, y, z normalizados).
            esquerda: ``(n,)``; ``True`` quando o frame é de mão esquerda.
            proporcao: Largura/altura da imagem de origem dos landmarks.

        Returns:
            ``(probabilidades, validos)``. ``validos`` tem tamanho ``n`` e marca os frames
            cujas características são finitas; ``probabilidades`` tem uma linha por frame
            válido (na mesma ordem) e uma coluna por classe, em [0, 1].
        """
        n = pontos.shape[0]
        vazio = np.empty((0, len(self.classes)), dtype=np.float64)
        if n == 0:
            return vazio, np.zeros(0, dtype=bool)

        with np.errstate(all="ignore"):   # landmarks degenerados geram NaN; filtrados abaixo
            caracteristicas = caracteristicas_mao.extrair(pontos, esquerda, proporcao)
        validos = np.isfinite(caracteristicas).all(axis=1)
        if not validos.any():
            return vazio, validos

        # float32 evita uma cópia: o scikit-learn converte as árvores para float32 de qualquer forma
        entrada = caracteristicas[validos].astype(np.float32, copy=False)
        probabilidades = np.asarray(self._estimador.predict_proba(entrada), dtype=np.float64)
        return probabilidades, validos


def carregar_modelo(caminho: str, sha256_esperado: Optional[str] = None) -> ModeloLibras:
    """Carrega e valida o pacote salvo por ``treinar_classificador.py``.

    Args:
        caminho: Arquivo ``.joblib``.
        sha256_esperado: Se informado, o SHA-256 do arquivo precisa ser igual (o joblib
            usa pickle, que executa código ao carregar).

    Returns:
        Modelo pronto para uso, já aquecido (a primeira inferência não paga o custo inicial).

    Raises:
        ErroModelo: Arquivo ausente, adulterado, no formato errado ou treinado com outras
            características.
    """
    if not os.path.isfile(caminho):
        raise ErroModelo(f"Modelo não encontrado em {caminho}. Treine primeiro: python treinar_classificador.py")

    if sha256_esperado is not None and calcular_sha256(caminho) != sha256_esperado.strip().lower():
        raise ErroModelo(f"SHA-256 de {caminho} diferente do esperado: arquivo alterado ou outro modelo.")

    try:
        pacote = joblib.load(caminho)
    except Exception as erro:  # joblib/pickle pode levantar quase qualquer exceção em arquivo ruim
        raise ErroModelo(f"Não foi possível ler {caminho}: {erro}") from erro

    if not isinstance(pacote, dict) or "modelo" not in pacote:
        raise ErroModelo(f"{caminho} não foi salvo por treinar_classificador.py.")
    if list(pacote.get("caracteristicas", [])) != caracteristicas_mao.NOMES:
        raise ErroModelo("O modelo foi treinado com outras características da mão. Treine de novo.")

    versao = pacote.get("versao_sklearn")
    if versao != sklearn.__version__:
        logger.warning("Modelo treinado com scikit-learn %s e este é o %s; se as previsões parecerem "
                       "estranhas, treine de novo.", versao, sklearn.__version__)

    estimador = pacote["modelo"]
    if hasattr(estimador, "n_jobs"):
        estimador.n_jobs = 1   # lotes pequenos: dividir as árvores entre processos custa mais do que economiza

    classes = tuple(str(c) for c in estimador.classes_)
    if ROTULO_BLANK in classes:
        raise ErroModelo(f"O modelo tem uma classe {ROTULO_BLANK!r}, reservada ao blank.")
    ignoradas = [c for c in classes if c not in ROTULOS_DO_FILTRO]
    if ignoradas:
        logger.warning("Classes que o filtro descartaria em silêncio (use a-z minúsculo e 0-9): %s", ignoradas)

    proporcao = float(pacote.get("proporcao_imagem", caracteristicas_mao.PROPORCAO_IMAGEM))
    modelo = ModeloLibras(estimador, classes, proporcao)

    # Aquecimento: a primeira chamada do scikit-learn é bem mais lenta que as seguintes
    estimador.predict_proba(np.zeros((1, len(caracteristicas_mao.NOMES)), dtype=np.float32))
    logger.info("Modelo carregado: %d classes (%s)", len(classes), " ".join(classes))
    return modelo


# =====================================================================
# DESSERIALIZAÇÃO E DECODIFICAÇÃO DO PACOTE
# =====================================================================
class _DesserializadorRestrito(pickle.Unpickler):
    """Unpickler que não aceita nenhum objeto global.

    O pacote do emissor só contém dict, list, str, int e float, tipos que o pickle
    reconstrói sem importar nada. Qualquer tentativa de carregar uma classe ou função
    (o caminho usado para executar código via pickle) é rejeitada.
    """

    def find_class(self, module: str, name: str) -> Any:
        raise pickle.UnpicklingError(f"objeto global proibido no pacote: {module}.{name}")


def desserializar_pacote(dados: bytes) -> Any:
    """Desserializa com segurança o pacote recebido do emissor.

    Args:
        dados: Bytes brutos de uma mensagem ZeroMQ.

    Returns:
        O objeto reconstruído (esperado: ``dict``).

    Raises:
        PacoteInvalido: Bytes corrompidos, truncados ou com objetos proibidos.
    """
    try:
        return _DesserializadorRestrito(io.BytesIO(dados)).load()
    except Exception as erro:  # entrada não confiável: o pickle pode falhar de muitas formas
        raise PacoteInvalido(f"pacote não desserializável ({type(erro).__name__}: {erro})") from erro


@dataclass(frozen=True)
class JanelaDecodificada:
    """Janela do emissor já convertida e validada.

    Attributes:
        sequencia: Número de sequência do emissor (``None`` se não enviado).
        frame_ids: ``(n,)`` identificadores crescentes dos frames.
        pontos: ``(n, 21, 3)`` landmarks (x, y, z).
        com_mao: ``(n,)``; ``True`` se o frame tem mão e landmarks finitos.
        esquerda: ``(n,)``; ``True`` se o frame é de mão esquerda.
        proporcao: Largura/altura da imagem.
        mao_informada: ``False`` se o pacote não trouxe ``maos`` (usou-se o padrão).
        proporcao_informada: ``False`` se o pacote não trouxe ``proporcao``.
    """

    sequencia: Optional[int]
    frame_ids: NDArray[np.int64]
    pontos: NDArray[np.float64]
    com_mao: NDArray[np.bool_]
    esquerda: NDArray[np.bool_]
    proporcao: float
    mao_informada: bool
    proporcao_informada: bool


@functools.lru_cache(maxsize=8)
def _resolver_colunas(colunas: tuple[str, ...]) -> tuple[int, NDArray[np.intp], int]:
    """Localiza, pelo nome, as colunas usadas (o layout do emissor pode mudar sem quebrar aqui).

    Args:
        colunas: Nomes das colunas na ordem das linhas de ``amostras``.

    Returns:
        ``(indice_frame_id, indices_xyz (21, 3), indice_confianca)``.

    Raises:
        PacoteInvalido: Falta alguma coluna obrigatória.
    """
    posicao = {nome: i for i, nome in enumerate(colunas)}
    try:
        indice_frame = posicao[COLUNA_FRAME]
        indice_confianca = posicao[COLUNA_CONFIANCA]
        xyz = np.array([[posicao[f"lm_{p}_{eixo}"] for eixo in "xyz"] for p in range(N_LANDMARKS)], dtype=np.intp)
    except KeyError as erro:
        raise PacoteInvalido(f"coluna obrigatória ausente: {erro}") from erro
    return indice_frame, xyz, indice_confianca


def _ler_lados(maos: Any, n: int, padrao_esquerda: bool) -> tuple[NDArray[np.bool_], bool]:
    """Converte a lista ``maos`` do pacote em um vetor booleano (``True`` = mão esquerda).

    Args:
        maos: ``None`` ou lista com ``n`` itens (``"Left"``, ``"Right"`` ou ``None``).
        n: Número de frames.
        padrao_esquerda: Lado assumido quando o nome não está disponível.

    Returns:
        ``(lados, informado)``.

    Raises:
        PacoteInvalido: ``maos`` não é lista ou tem tamanho diferente de ``n``.
    """
    lados = np.full(n, padrao_esquerda, dtype=bool)
    if maos is None:
        return lados, False
    if not isinstance(maos, (list, tuple)) or len(maos) != n:
        raise PacoteInvalido("campo 'maos' com tamanho diferente de 'amostras'")
    for i, nome in enumerate(maos):
        if nome in ("Left", "Right"):
            lados[i] = caracteristicas_mao.mao_esquerda(nome)
    return lados, True


def decodificar_pacote(pacote: Any, *, mao_padrao_esquerda: bool, proporcao_padrao: float) -> JanelaDecodificada:
    """Valida o pacote do emissor e o converte em arrays numpy.

    Args:
        pacote: Objeto devolvido por ``desserializar_pacote``.
        mao_padrao_esquerda: Lado assumido quando o pacote não informa ``maos``.
        proporcao_padrao: Proporção assumida quando o pacote não informa ``proporcao``.

    Returns:
        A janela decodificada.

    Raises:
        PacoteInvalido: Estrutura, tamanhos ou tipos fora do contrato.
    """
    if not isinstance(pacote, dict):
        raise PacoteInvalido("o pacote não é um dicionário")

    colunas, amostras = pacote.get("colunas"), pacote.get("amostras")
    if not isinstance(colunas, list) or not colunas or not all(isinstance(c, str) for c in colunas):
        raise PacoteInvalido("campo 'colunas' ausente ou malformado")
    if not isinstance(amostras, list) or not 0 < len(amostras) <= LIMITE_LINHAS_POR_PACOTE:
        raise PacoteInvalido(f"campo 'amostras' ausente ou fora de 1..{LIMITE_LINHAS_POR_PACOTE} linhas")

    indice_frame, xyz, indice_confianca = _resolver_colunas(tuple(colunas))
    try:
        matriz = np.asarray(amostras, dtype=np.float64)
    except (TypeError, ValueError) as erro:
        raise PacoteInvalido("'amostras' com linhas de tamanhos diferentes ou valores não numéricos") from erro
    if matriz.ndim != 2 or matriz.shape[1] != len(colunas):
        raise PacoteInvalido("'amostras' não tem uma coluna por nome em 'colunas'")

    ids_brutos = matriz[:, indice_frame]
    if not np.isfinite(ids_brutos).all():
        raise PacoteInvalido("'frame_id' com valores não finitos")

    pontos = matriz[:, xyz]                                  # (n, 21, 3)
    com_mao = (matriz[:, indice_confianca] > 0) & np.isfinite(pontos).all(axis=(1, 2))

    sequencia = pacote.get("sequencia")
    if sequencia is not None and (isinstance(sequencia, bool) or not isinstance(sequencia, int)):
        raise PacoteInvalido("'sequencia' não é inteiro")

    proporcao = pacote.get("proporcao")
    proporcao_informada = proporcao is not None
    if proporcao_informada:
        if (isinstance(proporcao, bool) or not isinstance(proporcao, (int, float))
                or not np.isfinite(proporcao) or not 0.1 <= proporcao <= 10.0):
            raise PacoteInvalido("'proporcao' inválida")
        proporcao = float(proporcao)
    else:
        proporcao = proporcao_padrao

    esquerda, mao_informada = _ler_lados(pacote.get("maos"), matriz.shape[0], mao_padrao_esquerda)
    return JanelaDecodificada(sequencia, ids_brutos.astype(np.int64), pontos, com_mao, esquerda,
                              proporcao, mao_informada, proporcao_informada)


# =====================================================================
# CLASSIFICAÇÃO
# =====================================================================
class RastreadorDeFrames:
    """Seleciona os frames ainda não classificados de cada janela.

    O emissor reenvia janelas de 120 frames a cada 15 a 45 frames, então cerca de 75% a
    90% de cada janela já foi vista. Classificar só o que é novo reduz o custo na mesma
    proporção e dá a cada frame exatamente uma previsão.
    """

    def __init__(self) -> None:
        """Inicia sem histórico."""
        self.ultimo_frame_id: Optional[int] = None
        self.ultima_sequencia: Optional[int] = None
        self.frames_perdidos: int = 0
        self.reinicios: int = 0

    def selecionar_novos(self, janela: JanelaDecodificada) -> NDArray[np.bool_]:
        """Marca os frames da janela que ainda não foram processados.

        Detecta o reinício do emissor (``sequencia`` que não cresce ou ``frame_id`` que
        volta) e recomeça a contagem; conta como perdidos os frames que ficaram para trás
        quando janelas inteiras se perdem.

        Args:
            janela: Janela decodificada.

        Returns:
            ``(n,)`` booleano; ``True`` nos frames novos.
        """
        ids = janela.frame_ids
        reiniciou = (
            (janela.sequencia is not None and self.ultima_sequencia is not None
             and janela.sequencia <= self.ultima_sequencia)
            or (self.ultimo_frame_id is not None and int(ids.max()) < self.ultimo_frame_id)
        )
        if reiniciou:
            self.ultimo_frame_id = None
            self.reinicios += 1
        if janela.sequencia is not None:
            self.ultima_sequencia = janela.sequencia

        if self.ultimo_frame_id is None:
            novos = np.ones(ids.shape, dtype=bool)
        else:
            novos = ids > self.ultimo_frame_id
        if novos.any():
            primeiro_novo = int(ids[novos].min())
            if self.ultimo_frame_id is not None and primeiro_novo > self.ultimo_frame_id + 1:
                self.frames_perdidos += primeiro_novo - self.ultimo_frame_id - 1
            self.ultimo_frame_id = int(ids.max())
        return novos


def montar_payload(
    classes: Sequence[str],
    massa_classes: NDArray[np.float64],
    massa_blank: float,
) -> dict[str, float]:
    """Monta o dicionário de probabilidades em porcentagem, no formato que o filtro lê.

    Args:
        classes: Rótulos, na ordem de ``massa_classes``.
        massa_classes: Fração (0 a 1) de cada classe; a soma com ``massa_blank`` é 1.
        massa_blank: Fração (0 a 1) do blank.

    Returns:
        ``{classe: porcentagem, ..., "-": porcentagem}``, com duas casas decimais.
    """
    payload = {c: round(float(m) * 100.0, 2) for c, m in zip(classes, massa_classes)}
    payload[ROTULO_BLANK] = round(float(massa_blank) * 100.0, 2)
    return payload


class PipelineClassificacao:
    """Transforma uma janela do emissor nos dicionários de probabilidade para o filtro."""

    def __init__(self, modelo: ModeloLibras, modo_saida: str) -> None:
        """Cria o pipeline.

        Args:
            modelo: Modelo carregado.
            modo_saida: ``"janela"`` (1 mensagem por janela) ou ``"frame"`` (1 por frame novo).

        Raises:
            ValueError: Modo desconhecido.
        """
        if modo_saida not in MODOS_SAIDA:
            raise ValueError(f"modo de saída desconhecido: {modo_saida!r}")
        self.modelo = modelo
        self.modo_saida = modo_saida
        self.rastreador = RastreadorDeFrames()
        self.frames_classificados = 0

    def processar(self, janela: JanelaDecodificada) -> list[dict[str, float]]:
        """Classifica os frames novos de uma janela.

        Args:
            janela: Janela decodificada.

        Returns:
            Zero (nada novo), uma (modo ``janela``) ou várias (modo ``frame``) previsões.
        """
        novos = np.flatnonzero(self.rastreador.selecionar_novos(janela))[-FRAMES_MAX_POR_PACOTE:]
        if novos.size == 0:
            return []
        self.frames_classificados += int(novos.size)

        com_mao = janela.com_mao[novos]
        com_mao_idx = np.flatnonzero(com_mao)
        probabilidades, validos = self.modelo.inferir(
            janela.pontos[novos][com_mao_idx], janela.esquerda[novos][com_mao_idx], janela.proporcao)

        valido_por_frame = np.zeros(novos.size, dtype=bool)
        valido_por_frame[com_mao_idx[validos]] = True
        return self._montar_saidas(valido_por_frame, probabilidades)

    def _montar_saidas(
        self,
        valido_por_frame: NDArray[np.bool_],
        probabilidades: NDArray[np.float64],
    ) -> list[dict[str, float]]:
        """Converte probabilidades por frame nas mensagens do modo de saída escolhido.

        Args:
            valido_por_frame: ``(n,)``; ``True`` nos frames com mão classificável.
            probabilidades: Uma linha por frame válido, na ordem dos frames.

        Returns:
            Lista de dicionários prontos para virar JSON.
        """
        classes = self.modelo.classes
        zeros = np.zeros(len(classes))

        if self.modo_saida == "frame":
            saidas: list[dict[str, float]] = []
            linha = 0
            for valido in valido_por_frame:
                if valido:
                    saidas.append(montar_payload(classes, probabilidades[linha], 0.0))
                    linha += 1
                else:
                    saidas.append(montar_payload(classes, zeros, 1.0))
            return saidas

        total = valido_por_frame.size
        com_mao = int(valido_por_frame.sum())
        fracao_blank = (total - com_mao) / total
        media = probabilidades.mean(axis=0) if com_mao else zeros
        return [montar_payload(classes, (1.0 - fracao_blank) * media, fracao_blank)]


# =====================================================================
# SERVIÇO ZEROMQ
# =====================================================================
@dataclass(frozen=True)
class Configuracao:
    """Parâmetros de execução.

    Attributes:
        caminho_modelo: Arquivo ``.joblib`` do Random Forest.
        sha256_modelo: SHA-256 esperado do modelo (``None`` = não conferir).
        endereco_entrada: Endpoint do ``bind`` PULL (emissor).
        endereco_saida: Endpoint do ``connect`` PUSH (filtro).
        modo_saida: ``"janela"`` ou ``"frame"``.
        mao_padrao_esquerda: Lado assumido quando o pacote não traz ``maos``.
    """

    caminho_modelo: str
    sha256_modelo: Optional[str]
    endereco_entrada: str
    endereco_saida: str
    modo_saida: str
    mao_padrao_esquerda: bool


@dataclass
class Estatisticas:
    """Contadores de operação, todos de tamanho fixo (não crescem com o tempo)."""

    pacotes: int = 0
    pulados_por_atraso: int = 0
    invalidos: int = 0
    erros_inesperados: int = 0
    mensagens_enviadas: int = 0
    mensagens_descartadas: int = 0
    latencia_media_ms: float = 0.0
    latencia_max_ms: float = 0.0

    def registrar_latencia(self, ms: float) -> None:
        """Atualiza a média móvel exponencial (alfa 0,1) e o máximo do período.

        Args:
            ms: Tempo gasto em desserializar, classificar e enviar um pacote.
        """
        self.latencia_media_ms = ms if self.latencia_media_ms == 0.0 else 0.9 * self.latencia_media_ms + 0.1 * ms
        self.latencia_max_ms = max(self.latencia_max_ms, ms)


class ServidorClassificador:
    """Laço PULL -> classificação -> PUSH, em uma única thread e sem chamadas bloqueantes."""

    def __init__(self, configuracao: Configuracao, modelo: ModeloLibras) -> None:
        """Cria o servidor (os sockets só são abertos em ``executar``).

        Args:
            configuracao: Parâmetros de execução.
            modelo: Modelo já carregado.
        """
        self._cfg = configuracao
        self._pipeline = PipelineClassificacao(modelo, configuracao.modo_saida)
        self.estatisticas = Estatisticas()
        self._contexto: Optional[zmq.Context] = None
        self._entrada: Optional[zmq.Socket] = None
        self._saida: Optional[zmq.Socket] = None
        self._parar = False
        self._avisou_mao_padrao = False
        self._avisou_proporcao_padrao = False

    # ---- ciclo de vida -------------------------------------------------
    def executar(self) -> int:
        """Abre os sockets, atende até receber SIGINT/SIGTERM e libera os recursos.

        Returns:
            Código de saída: 0 (encerrado normalmente) ou 3 (não conseguiu abrir os sockets).
        """
        self._instalar_sinais()
        try:
            self._abrir_sockets()
        except zmq.ZMQError as erro:
            if erro.errno == errno.EADDRINUSE:
                logger.error("Endereço %s já está em uso (há outro classificador rodando?).",
                             self._cfg.endereco_entrada)
            else:
                logger.error("Falha ao abrir os sockets: %s", erro)
            self._fechar()
            return 3

        logger.info("Pronto. Entrada (bind PULL): %s | Saída (connect PUSH): %s | modo de saída: %s",
                    self._cfg.endereco_entrada, self._cfg.endereco_saida, self._cfg.modo_saida)
        try:
            self._laco_principal()
        finally:
            self._registrar_resumo("Resumo final")
            self._fechar()
        return 0

    def _instalar_sinais(self) -> None:
        """Faz SIGINT e SIGTERM encerrarem o laço de forma ordenada (o poll acorda em até 0,5 s)."""
        def tratador(numero: int, _quadro: Optional[FrameType]) -> None:
            logger.info("Sinal %d recebido: encerrando.", numero)
            self._parar = True

        for sinal in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sinal, tratador)
            except ValueError:   # fora da thread principal não é possível instalar tratadores
                logger.debug("Tratador do sinal %s não instalado (fora da thread principal).", sinal)

    def _abrir_sockets(self) -> None:
        """Cria o PULL (``bind``) e o PUSH (``connect``) com limites de fila e de tamanho.

        Raises:
            zmq.ZMQError: Endereço em uso ou inválido.
        """
        self._contexto = zmq.Context()

        self._entrada = self._contexto.socket(zmq.PULL)
        self._entrada.setsockopt(zmq.LINGER, 0)
        self._entrada.setsockopt(zmq.RCVHWM, HWM_ENTRADA)
        self._entrada.setsockopt(zmq.MAXMSGSIZE, TAMANHO_MAXIMO_PACOTE_BYTES)   # evita esgotar memória
        self._ativar_heartbeat(self._entrada)
        self._entrada.bind(self._cfg.endereco_entrada)

        self._saida = self._contexto.socket(zmq.PUSH)
        self._saida.setsockopt(zmq.LINGER, 1000)        # ao encerrar, tenta entregar o que sobrou por até 1 s
        self._saida.setsockopt(zmq.SNDHWM, HWM_SAIDA)
        self._saida.setsockopt(zmq.IMMEDIATE, 1)        # sem filtro conectado, não acumula previsões velhas
        self._ativar_heartbeat(self._saida)
        self._saida.connect(self._cfg.endereco_saida)

    @staticmethod
    def _ativar_heartbeat(socket: zmq.Socket) -> None:
        """Liga o heartbeat do ZeroMQ para detectar pares mortos (precisa de libzmq >= 4.2).

        Args:
            socket: Socket a configurar; versões sem suporte são ignoradas.
        """
        try:
            socket.setsockopt(zmq.HEARTBEAT_IVL, 1000)
            socket.setsockopt(zmq.HEARTBEAT_TIMEOUT, 3000)
        except (AttributeError, zmq.ZMQError):
            logger.debug("Heartbeat do ZeroMQ indisponível nesta versão.")

    def _fechar(self) -> None:
        """Fecha os sockets e encerra o contexto (idempotente)."""
        if self._entrada is not None:
            self._entrada.close(linger=0)
        if self._saida is not None:
            self._saida.close(linger=1000)
        if self._contexto is not None:
            self._contexto.term()
        self._entrada = self._saida = self._contexto = None

    # ---- laço ----------------------------------------------------------
    def _laco_principal(self) -> None:
        """Espera pacotes com timeout, processa o mais recente e emite o resumo periódico."""
        assert self._entrada is not None
        poller = zmq.Poller()
        poller.register(self._entrada, zmq.POLLIN)

        inicio = time.monotonic()
        ultimo_pacote_em = inicio
        ultimo_resumo_em = inicio
        avisou_ociosidade = False

        while not self._parar:
            try:
                prontos = dict(poller.poll(TIMEOUT_POLL_MS))
            except zmq.ZMQError as erro:
                if erro.errno == errno.EINTR:
                    continue
                raise

            agora = time.monotonic()
            if self._entrada in prontos:
                recebido = self._receber_mais_recente()
                if recebido is not None:
                    bruto, pulados = recebido
                    self.estatisticas.pulados_por_atraso += pulados
                    ultimo_pacote_em, avisou_ociosidade = agora, False
                    self._tratar_pacote(bruto)
            elif not avisou_ociosidade and agora - ultimo_pacote_em >= SEM_PACOTES_AVISO_S:
                logger.warning("Nenhum pacote do emissor há %.0f s (emissor parado ou em outro endereço?).",
                               agora - ultimo_pacote_em)
                avisou_ociosidade = True

            if agora - ultimo_resumo_em >= INTERVALO_RESUMO_S:
                self._registrar_resumo("Resumo")
                ultimo_resumo_em = agora

    def _receber_mais_recente(self) -> Optional[tuple[bytes, int]]:
        """Lê tudo o que está pendente e fica só com a mensagem mais nova.

        Se o classificador atrasar, janelas velhas só aumentariam a latência. Como cada
        janela repete os frames das anteriores, pular uma quase nunca perde informação.
        As mensagens puladas nem são desserializadas.

        Returns:
            ``(bytes_da_mensagem_mais_nova, quantidade_pulada)`` ou ``None`` se não havia nada.
        """
        assert self._entrada is not None
        try:
            ultima = self._entrada.recv(flags=zmq.NOBLOCK)
        except zmq.Again:
            return None
        puladas = 0
        while True:
            try:
                proxima = self._entrada.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                return ultima, puladas
            ultima, puladas = proxima, puladas + 1

    def _tratar_pacote(self, bruto: bytes) -> None:
        """Desserializa, classifica e envia; nenhum erro de um pacote derruba o serviço.

        Args:
            bruto: Bytes de uma mensagem do emissor.
        """
        inicio = time.perf_counter()
        self.estatisticas.pacotes += 1
        try:
            pacote = desserializar_pacote(bruto)
            janela = decodificar_pacote(
                pacote,
                mao_padrao_esquerda=self._cfg.mao_padrao_esquerda,
                proporcao_padrao=self._pipeline.modelo.proporcao_treino,
            )
            self._avisar_campos_ausentes(janela)
            saidas = self._pipeline.processar(janela)
        except PacoteInvalido as erro:
            self.estatisticas.invalidos += 1
            logger.warning("Pacote descartado: %s", erro)
            return
        except Exception:  # noqa: BLE001 - falha imprevista de inferência não pode matar o processo
            self.estatisticas.erros_inesperados += 1
            logger.exception("Falha inesperada ao processar o pacote; ele foi descartado.")
            return

        for saida in saidas:
            self._enviar(saida)
        self.estatisticas.registrar_latencia((time.perf_counter() - inicio) * 1000.0)

    def _avisar_campos_ausentes(self, janela: JanelaDecodificada) -> None:
        """Avisa uma única vez quando o emissor não manda ``maos`` ou ``proporcao``.

        Args:
            janela: Janela recém-decodificada.
        """
        if not janela.mao_informada and not self._avisou_mao_padrao:
            lado = "esquerda" if self._cfg.mao_padrao_esquerda else "direita"
            logger.warning("O pacote não traz 'maos': assumindo mão %s em todos os frames. "
                           "Com a outra mão a acurácia cai.", lado)
            self._avisou_mao_padrao = True
        if not janela.proporcao_informada and not self._avisou_proporcao_padrao:
            logger.warning("O pacote não traz 'proporcao': usando %.3f (a do treino). Se a câmera do emissor "
                           "não for 640x480, as distâncias saem distorcidas.", janela.proporcao)
            self._avisou_proporcao_padrao = True

    def _enviar(self, payload: dict[str, float]) -> None:
        """Envia uma previsão ao filtro sem nunca bloquear.

        Args:
            payload: Dicionário classe -> porcentagem (inclui ``"-"``).
        """
        assert self._saida is not None
        try:
            corpo = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
            self._saida.send(corpo, flags=zmq.NOBLOCK)
            self.estatisticas.mensagens_enviadas += 1
        except zmq.Again:
            # Filtro desconectado ou fila cheia: a previsão fica velha rápido, então é descartada
            self.estatisticas.mensagens_descartadas += 1
            logger.debug("Filtro indisponível: previsão descartada.")
        except (zmq.ZMQError, ValueError) as erro:
            self.estatisticas.mensagens_descartadas += 1
            logger.warning("Falha ao enviar ao filtro: %s", erro)

    def _registrar_resumo(self, titulo: str) -> None:
        """Escreve os contadores no log e reinicia o máximo de latência do período.

        Args:
            titulo: Prefixo da linha de log.
        """
        e, r = self.estatisticas, self._pipeline.rastreador
        logger.info(
            "%s: pacotes=%d pulados=%d invalidos=%d erros=%d | frames_classificados=%d perdidos=%d "
            "reinicios_emissor=%d | enviadas=%d descartadas=%d | latencia media=%.1f ms max=%.1f ms",
            titulo, e.pacotes, e.pulados_por_atraso, e.invalidos, e.erros_inesperados,
            self._pipeline.frames_classificados, r.frames_perdidos, r.reinicios,
            e.mensagens_enviadas, e.mensagens_descartadas, e.latencia_media_ms, e.latencia_max_ms)
        if e.latencia_max_ms > LIMITE_LATENCIA_MS:
            logger.warning("Latência máxima de %.0f ms acima de %.0f ms: o classificador pode não acompanhar "
                           "o emissor.", e.latencia_max_ms, LIMITE_LATENCIA_MS)
        e.latencia_max_ms = 0.0


# =====================================================================
# LINHA DE COMANDO
# =====================================================================
def _ler_argumentos(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    """Lê os argumentos da linha de comando.

    Args:
        argv: Argumentos (``None`` usa ``sys.argv``).

    Returns:
        Argumentos interpretados.
    """
    p = argparse.ArgumentParser(description="Classificador ZeroMQ de letras em tempo real (Libras).")
    p.add_argument("--modelo", default=CAMINHO_MODELO_PADRAO, help="arquivo .joblib salvo pelo treino")
    p.add_argument("--modelo-sha256", default=None, help="SHA-256 esperado do modelo (opcional)")
    p.add_argument("--endereco-entrada", default=ENDERECO_ENTRADA_PADRAO, help="bind PULL (emissor)")
    p.add_argument("--endereco-saida", default=ENDERECO_SAIDA_PADRAO, help="connect PUSH (filtro)")
    p.add_argument("--modo-saida", choices=MODOS_SAIDA, default="janela",
                   help="janela: 1 mensagem por janela | frame: 1 mensagem por frame novo")
    p.add_argument("--mao-padrao", choices=("direita", "esquerda"), default="direita",
                   help="mão assumida se o emissor não enviar 'maos'")
    p.add_argument("--log-nivel", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Ponto de entrada.

    Args:
        argv: Argumentos da linha de comando (``None`` usa ``sys.argv``).

    Returns:
        Código de saída do processo: 0 (ok), 2 (modelo inválido) ou 3 (sockets).
    """
    args = _ler_argumentos(argv)
    logging.basicConfig(level=args.log_nivel, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    try:
        modelo = carregar_modelo(args.modelo, args.modelo_sha256)
    except ErroModelo as erro:
        logger.error("%s", erro)
        return 2

    configuracao = Configuracao(
        caminho_modelo=args.modelo,
        sha256_modelo=args.modelo_sha256,
        endereco_entrada=args.endereco_entrada,
        endereco_saida=args.endereco_saida,
        modo_saida=args.modo_saida,
        mao_padrao_esquerda=args.mao_padrao == "esquerda",
    )
    return ServidorClassificador(configuracao, modelo).executar()


if __name__ == "__main__":
    sys.exit(main())