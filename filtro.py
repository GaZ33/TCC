"""
filtro_1.py: filtro entre o classificador e o exibidor.

  Classificador --(5556, JSON)--> FILTRO --(5557, JSON)--> Exibidor

Recebe do classificador um dicionário de probabilidades por classe, escolhe a melhor e
decide o que o exibidor mostra:
  * a melhor classe é uma letra/número com confiança >= LIMIAR_CONFIANCA  -> SHOW
  * a melhor classe é "-" (sem mão) OU a confiança é menor que o limiar    -> blank (nada é enviado)

A confiança é a fatia da melhor classe no total, em %, então não depende de o classificador
mandar 0 a 1 ou 0 a 100.
"""

import json
import logging
import math

import zmq


# ============================================================
# 1. CONFIGURAÇÕES
# ============================================================

ENDPOINT_CLASSIFICADOR = "tcp://127.0.0.1:5556"   # bind (PULL): o classificador conecta aqui
ENDPOINT_EXIBIDOR = "tcp://127.0.0.1:5557"        # connect (PUSH): o exibidor faz bind aqui

# O classificador deve usar "-" para representar blank.
BLANK_LABEL = "-"
LABELS = [BLANK_LABEL] + list("abcdefghijklmnopqrstuvwxyz0123456789")

# Confiança mínima (em %) da melhor classe para virar SHOW. Abaixo disso a previsão conta
# como blank. 50 é um ponto de partida: ajuste olhando as probabilidades reais (nível DEBUG).
LIMIAR_CONFIANCA = 50.0

# Blanks consecutivos para mandar um CLEAR ao exibidor. 0 = desligado (o exibidor atual
# ignora CLEAR; deixe 0 para não enviar eventos que ninguém usa).
BLANKS_PARA_LIMPAR = 0

# Chaves aceitas caso o classificador envie a previsão dentro de um envelope.
ENVELOPES = ("probabilidades", "predicao", "probabilities")

LOG_NIVEL = logging.INFO


# ============================================================
# 2. LEITURA E VALIDAÇÃO DA PREVISÃO
# ============================================================

def extrair_probabilidades(payload):
    """
    Aceita uma única previsão nestes formatos:
      {"a": 70, "b": 25, "-": 5}
      {"probabilidades": {"a": 70, "b": 25, "-": 5}}
      [["a", 70], ["b", 25], ["-", 5]]

    Não processa uma lista de várias previsões de uma só vez.
    """

    if isinstance(payload, dict):
        for chave in ENVELOPES:
            if chave in payload:
                payload = payload[chave]
                break

    if isinstance(payload, dict):
        probabilidades = payload

    elif isinstance(payload, (list, tuple)) and payload:
        probabilidades = {}

        for item in payload:
            if (
                not isinstance(item, (list, tuple))
                or len(item) != 2
                or not isinstance(item[0], str)
            ):
                raise ValueError(
                    "Formato inválido: esperado uma lista de pares (classe, percentual)."
                )

            classe, valor = item
            if classe in probabilidades:
                raise ValueError(f"Classe repetida na previsão: {classe!r}.")
            probabilidades[classe] = valor
    else:
        raise ValueError("Formato da previsão não reconhecido.")

    # Aceita "blank" como nome alternativo para a classe "-".
    if "blank" in probabilidades and BLANK_LABEL not in probabilidades:
        probabilidades[BLANK_LABEL] = probabilidades["blank"]

    valores = {}
    for classe, valor in probabilidades.items():
        if classe not in LABELS and classe != "blank":
            continue

        try:
            valor = float(valor)
        except (TypeError, ValueError) as erro:
            raise ValueError(f"Valor inválido para a classe {classe!r}.") from erro

        if not math.isfinite(valor) or valor < 0:
            raise ValueError(
                f"O valor da classe {classe!r} deve ser finito e não negativo."
            )

        classe_normalizada = BLANK_LABEL if classe == "blank" else classe
        valores[classe_normalizada] = valor

    if not valores or max(valores.values()) <= 0:
        raise ValueError("A previsão não contém estimativas positivas válidas.")

    return valores


def escolher_classe(probabilidades):
    """Retorna a classe com a maior estimativa."""
    return max(probabilidades, key=probabilidades.get)


def calcular_confianca(probabilidades, classe):
    """Fatia da classe no total das estimativas, em % (0 a 100)."""
    return 100.0 * probabilidades[classe] / sum(probabilidades.values())


# ============================================================
# 3. LÓGICA DO FILTRO
# ============================================================

class Filtro:
    def __init__(self):
        self.blanks_consecutivos = 0
        self.tela_limpa = False
        self.abaixo_do_limiar = 0     # quantas previsões viraram blank por confiança baixa

    def processar_previsao(self, probabilidades):
        classe = escolher_classe(probabilidades)
        confianca = calcular_confianca(probabilidades, classe)

        sem_mao = classe == BLANK_LABEL
        incerta = not sem_mao and confianca < LIMIAR_CONFIANCA
        if incerta:
            self.abaixo_do_limiar += 1

        if sem_mao or incerta:
            self.blanks_consecutivos += 1
            logging.debug(
                "Blank (%s) consecutivo: %d | melhor: %s com %.1f%%",
                "sem mão" if sem_mao else "confiança baixa",
                self.blanks_consecutivos,
                classe,
                confianca,
            )

            # Envia CLEAR somente uma vez por sequência de blanks (e só se estiver ligado).
            if (
                BLANKS_PARA_LIMPAR > 0
                and self.blanks_consecutivos >= BLANKS_PARA_LIMPAR
                and not self.tela_limpa
            ):
                self.tela_limpa = True
                return {"acao": "CLEAR", "valor": "space"}

            return None

        # Uma letra ou número confiável interrompe a sequência de blanks.
        self.blanks_consecutivos = 0
        self.tela_limpa = False

        # Cada previsão vencedora e confiável é encaminhada ao exibidor.
        return {"acao": "SHOW", "valor": classe}


# ============================================================
# 4. COMUNICAÇÃO ZEROMQ
# ============================================================

def main():
    logging.basicConfig(
        level=LOG_NIVEL,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    contexto = zmq.Context.instance()

    socket_entrada = contexto.socket(zmq.PULL)
    socket_entrada.setsockopt(zmq.LINGER, 0)
    socket_entrada.setsockopt(zmq.RCVTIMEO, 500)
    socket_entrada.bind(ENDPOINT_CLASSIFICADOR)

    socket_saida = contexto.socket(zmq.PUSH)
    socket_saida.setsockopt(zmq.LINGER, 0)
    socket_saida.connect(ENDPOINT_EXIBIDOR)

    filtro = Filtro()
    recebidas = 0
    invalidas = 0
    eventos_enviados = 0

    logging.info("Filtro iniciado.")
    logging.info("Entrada do classificador: %s", ENDPOINT_CLASSIFICADOR)
    logging.info("Saída para o exibidor: %s", ENDPOINT_EXIBIDOR)
    logging.info("Confiança mínima para mostrar: %.0f%%", LIMIAR_CONFIANCA)
    if BLANKS_PARA_LIMPAR > 0:
        logging.info("CLEAR após %d blanks consecutivos.", BLANKS_PARA_LIMPAR)

    try:
        while True:
            try:
                bruto = socket_entrada.recv()
            except zmq.Again:
                # Permite encerrar com Ctrl+C mesmo sem mensagens chegando.
                continue

            recebidas += 1

            try:
                payload = json.loads(bruto.decode("utf-8"))
                probabilidades = extrair_probabilidades(payload)
                evento = filtro.processar_previsao(probabilidades)
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as erro:
                invalidas += 1
                logging.warning("Previsão inválida: %s", erro)
                continue

            classe = escolher_classe(probabilidades)
            logging.debug("Previsão recebida: %s", probabilidades)

            if evento is not None:
                socket_saida.send_json(evento)
                eventos_enviados += 1
                logging.info("Classe vencedora: %s (%.0f%%) | Evento enviado: %s",
                             classe, calcular_confianca(probabilidades, classe), evento)

    except KeyboardInterrupt:
        logging.info("Encerrando filtro...")

    finally:
        logging.info(
            "Resumo: recebidas=%d inválidas=%d eventos_enviados=%d abaixo_do_limiar=%d",
            recebidas,
            invalidas,
            eventos_enviados,
            filtro.abaixo_do_limiar,
        )
        socket_entrada.close()
        socket_saida.close()


if __name__ == "__main__":
    main()