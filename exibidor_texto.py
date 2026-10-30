"""
exibidor_texto.py: final da cadeia (Emissor -> Classificador -> Filtro -> EXIBIDOR).

Mostra para o usuário a imagem da câmera (enviada pelo emissor) e a letra ou o número
reconhecido. Não abre a câmera: ela fica com o emissor, e as imagens chegam por ZeroMQ.

Sockets (PULL, este programa faz ``bind``):
  tcp://127.0.0.1:5557  eventos do filtro: {"acao": "SHOW", "valor": "a"}
  tcp://127.0.0.1:5558  imagens JPEG da câmera, enviadas pelo emissor (opcional)

Modos (a tecla M troca, com o programa rodando):
  MANUAL       o primeiro resultado fica congelado até o usuário apertar a tecla de reinício
  AUTOMATICO   cada resultado novo substitui o anterior, e o último fica na tela até chegar
               outro ou o usuário limpar

Nos dois modos, o evento CLEAR do filtro é ignorado (nada apaga a tela sozinho).

Teclas (clique na janela antes):
  M troca o modo | T troca a tecla de reinício | ESPACO (ou C) limpa | V liga/desliga a
  câmera | Q sai
"""

import queue
import threading
import time
import cv2
import numpy as np
import zmq

ENDPOINT_EXIBIDOR = "tcp://127.0.0.1:5557"   # eventos do filtro
ENDPOINT_CAMERA = "tcp://127.0.0.1:5558"     # imagens do emissor

# =====================================================================
# CONFIGURAÇÕES (ajuste aqui)
# =====================================================================
MODO_MANUAL = "MANUAL"
MODO_AUTOMATICO = "AUTOMATICO"
MODO_PADRAO = MODO_MANUAL
TECLA_REINICIO_PADRAO = " "        # " " = espaço | "\r" = Enter | ou qualquer tecla comum (ex.: "r")
MOSTRAR_CAMERA_PADRAO = True
ACEITAR_CLEAR_DO_FILTRO = False    # False: o CLEAR (blanks) não apaga a tela

# Teclas fixas (não podem ser escolhidas como tecla de reinício)
TECLA_SAIR = "q"
TECLA_MODO = "m"
TECLA_TROCAR = "t"
TECLA_LIMPAR = "c"
TECLA_CAMERA = "v"
TECLA_ESC = 27
TECLAS_RESERVADAS = {ord(TECLA_SAIR), ord(TECLA_MODO), ord(TECLA_TROCAR),
                     ord(TECLA_LIMPAR), ord(TECLA_CAMERA), TECLA_ESC}

# --- Tela ---
LARGURA = 640
ALTURA_PAINEL = 200                # painel do texto
CAMERA_ALTURA = 480                # área da imagem da câmera (acima do painel)
CAMERA_TIMEOUT_S = 2.0             # sem imagem há tanto tempo: mostra "Sem imagem"
TAMANHO_MAXIMO_IMAGEM = 4 * 1024 * 1024

FONTE = cv2.FONT_HERSHEY_SIMPLEX
COR_TEXTO = (0, 255, 0)
COR_DICA = (160, 160, 160)
ESCALA_MAX = 2.2
ESCALA_RODAPE = 0.5
MARGEM_X = 30
AVISO_SEGUNDOS = 2.5

# A letra "O" e o número "0" têm sinais idênticos: não dá para distinguir um do outro.
# (Sem acentos de propósito: as fontes do OpenCV não desenham acentos.)
LINHAS_AMBIGUO = ["Letra O ou numero 0.", "Sinais identicos"]


# =====================================================================
# TECLAS
# =====================================================================
def normalizar_tecla(tecla):
    """Letras maiúsculas viram minúsculas (Shift/Caps Lock não mudam o comando)."""
    return tecla + 32 if 65 <= tecla <= 90 else tecla


def tecla_valida(tecla):
    """Enter ou qualquer caractere imprimível (inclui espaço)."""
    return tecla == 13 or 32 <= tecla <= 126


def tecla_aceita(tecla):
    return tecla_valida(tecla) and tecla not in TECLAS_RESERVADAS


def nome_tecla(tecla):
    if tecla == 32:
        return "ESPACO"
    if tecla == 13:
        return "ENTER"
    return chr(tecla).upper()


# =====================================================================
# ESTADO
# =====================================================================
def esvaziar(fila):
    """Joga fora tudo o que está na fila (eventos velhos)."""
    try:
        while True:
            fila.get_nowait()
    except queue.Empty:
        pass


class Estado:
    def __init__(self):
        self.modo = MODO_PADRAO
        self.tecla_reinicio = ord(TECLA_REINICIO_PADRAO)
        if not tecla_aceita(self.tecla_reinicio):
            print("[AVISO] TECLA_REINICIO_PADRAO inválida ou reservada; usando ESPACO.")
            self.tecla_reinicio = 32
        self.valor = "-"             # última letra ou número recebido ("-" = nada)
        self.travado = False         # True = resultado congelado (só no modo MANUAL)
        self.trocando_tecla = False  # True = esperando o usuário apertar a nova tecla
        self.mostrar_camera = MOSTRAR_CAMERA_PADRAO
        self.aviso = ""
        self.aviso_ate = 0.0

    def avisar(self, texto):
        self.aviso = texto
        self.aviso_ate = time.monotonic() + AVISO_SEGUNDOS

    def reiniciar(self, fila):
        self.valor = "-"
        self.travado = False
        esvaziar(fila)    # o que chegou durante o congelamento não deve reaparecer


def processar_eventos(fila, estado):
    """Lê os eventos que chegaram.
    MANUAL: o primeiro SHOW congela o resultado; depois disso todo evento é descartado.
    AUTOMATICO: cada SHOW substitui o resultado; o último fica na tela até chegar outro.
    O CLEAR do filtro é ignorado, a menos que ACEITAR_CLEAR_DO_FILTRO esteja ligado."""
    while True:
        try:
            evento = fila.get_nowait()
        except queue.Empty:
            break

        if estado.travado or not isinstance(evento, dict):
            continue

        acao = evento.get("acao")
        valor = evento.get("valor")

        if acao == "SHOW" and isinstance(valor, str) and valor:
            estado.valor = valor.upper()
            if estado.modo == MODO_MANUAL:
                estado.travado = True
        elif acao == "CLEAR" and ACEITAR_CLEAR_DO_FILTRO:
            estado.valor = "-"


def tratar_tecla(estado, tecla, fila):
    """Executa o comando da tecla apertada. Retorna True se for para sair."""
    if tecla == 255:      # waitKey devolve 255 (-1 & 0xFF) quando nenhuma tecla foi apertada
        return False
    tecla = normalizar_tecla(tecla)

    # Esperando a nova tecla de reinício: a próxima tecla não executa comando nenhum
    if estado.trocando_tecla:
        if tecla == TECLA_ESC:
            estado.trocando_tecla = False
            estado.avisar("Troca cancelada")
        elif not tecla_aceita(tecla):
            estado.avisar("Essa tecla nao pode ser usada. Escolha outra.")
        else:
            estado.tecla_reinicio = tecla
            estado.trocando_tecla = False
            estado.avisar(f"Tecla de reinicio agora: {nome_tecla(tecla)}")
        return False

    if tecla == ord(TECLA_SAIR):
        return True
    if tecla == ord(TECLA_MODO):
        estado.modo = MODO_AUTOMATICO if estado.modo == MODO_MANUAL else MODO_MANUAL
        estado.reiniciar(fila)            # trocar de modo recomeça com a tela limpa
        estado.avisar(f"Modo {estado.modo}")
    elif tecla == ord(TECLA_TROCAR):
        estado.trocando_tecla = True
        estado.aviso_ate = 0.0            # apaga um aviso antigo para o pedido aparecer na hora
    elif tecla == ord(TECLA_CAMERA):
        estado.mostrar_camera = not estado.mostrar_camera
        estado.avisar("Camera ligada" if estado.mostrar_camera else "Camera desligada")
    elif tecla in (estado.tecla_reinicio, ord(TECLA_LIMPAR)):
        estado.reiniciar(fila)
    return False


# =====================================================================
# DESENHO
# =====================================================================
def montar_linhas(valor):
    """Decide o que escrever na tela. 'valor' já vem em maiúsculo ('-' = nada ainda)."""
    if valor in ("O", "0"):
        return LINHAS_AMBIGUO
    if valor.isdigit():
        return [f"Numero: {valor}"]
    return [f"Letra: {valor}"]


def desenhar_linhas(frame, linhas):
    """Escreve as linhas centralizadas na vertical, encolhendo o texto se não couber na largura."""
    altura, largura = frame.shape[:2]

    # Mede a linha mais larga no tamanho máximo e reduz a escala só se passar da janela
    mais_larga = max(cv2.getTextSize(l, FONTE, ESCALA_MAX, 3)[0][0] for l in linhas)
    escala = min(ESCALA_MAX, ESCALA_MAX * (largura - 2 * MARGEM_X) / mais_larga)
    espessura = 3 if escala >= 1.5 else 2

    alturas = [cv2.getTextSize(l, FONTE, escala, espessura)[0][1] for l in linhas]
    espaco = int(0.6 * max(alturas))                       # espaço entre as linhas
    total = sum(alturas) + espaco * (len(linhas) - 1)

    y = (altura - total) // 2
    for linha, h in zip(linhas, alturas):
        y += h                                             # y = base da linha
        cv2.putText(frame, linha, (MARGEM_X, y), FONTE, escala, COR_TEXTO, espessura, cv2.LINE_AA)
        y += espaco


def textos_rodape(estado, agora):
    """As duas linhas pequenas do rodapé: situação atual e atalhos."""
    nome = nome_tecla(estado.tecla_reinicio)
    if agora < estado.aviso_ate:
        linha1 = estado.aviso
    elif estado.trocando_tecla:
        linha1 = "Aperte a nova tecla de reinicio (ESC cancela)"
    elif estado.modo == MODO_AUTOMATICO:
        linha1 = f"AUTOMATICO: mantem o ultimo; {nome} limpa"
    elif estado.travado:
        linha1 = f"CONGELADO: aperte {nome} para reiniciar"
    else:
        linha1 = f"MANUAL: aguardando; depois {nome} reinicia"
    linha2 = (f"{TECLA_MODO}: modo | {TECLA_TROCAR}: trocar tecla | "
              f"{TECLA_CAMERA}: camera | {TECLA_SAIR}: sair")
    return linha1, linha2


def desenhar_rodape(frame, estado):
    altura = frame.shape[0]
    linha1, linha2 = textos_rodape(estado, time.monotonic())
    cv2.putText(frame, linha1, (MARGEM_X, altura - 28), FONTE, ESCALA_RODAPE, COR_DICA, 1, cv2.LINE_AA)
    cv2.putText(frame, linha2, (MARGEM_X, altura - 10), FONTE, ESCALA_RODAPE, COR_DICA, 1, cv2.LINE_AA)


def desenhar_camera(imagem):
    """Área da câmera: a imagem ajustada (sem deformar) ou um aviso se não há imagem."""
    area = np.zeros((CAMERA_ALTURA, LARGURA, 3), dtype=np.uint8)
    if imagem is None:
        texto = "Sem imagem da camera"
        (w, h), _ = cv2.getTextSize(texto, FONTE, 0.8, 2)
        cv2.putText(area, texto, ((LARGURA - w) // 2, (CAMERA_ALTURA + h) // 2),
                    FONTE, 0.8, COR_DICA, 2, cv2.LINE_AA)
        return area

    h, w = imagem.shape[:2]
    escala = min(LARGURA / w, CAMERA_ALTURA / h)
    novo_w, novo_h = max(1, int(w * escala)), max(1, int(h * escala))
    ajustada = cv2.resize(imagem, (novo_w, novo_h))
    x0, y0 = (LARGURA - novo_w) // 2, (CAMERA_ALTURA - novo_h) // 2
    area[y0:y0 + novo_h, x0:x0 + novo_w] = ajustada
    return area


# =====================================================================
# ZEROMQ
# =====================================================================
class PreviaCamera:
    """Guarda a imagem mais recente recebida (escrita pela thread de rede, lida pela principal)."""

    def __init__(self):
        self._trava = threading.Lock()
        self._imagem = None
        self._recebida_em = 0.0

    def atualizar(self, imagem):
        with self._trava:
            self._imagem = imagem
            self._recebida_em = time.monotonic()

    def obter(self, agora):
        """Retorna a imagem mais recente, ou None se não há ou se está velha demais."""
        with self._trava:
            if self._imagem is None or agora - self._recebida_em > CAMERA_TIMEOUT_S:
                return None
            return self._imagem


def ouvir_zmq(fila):
    contexto = zmq.Context.instance()
    socket = contexto.socket(zmq.PULL)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(ENDPOINT_EXIBIDOR)
    print(f"[ZMQ] Ouvindo eventos em {ENDPOINT_EXIBIDOR}")
    try:
        while True:
            evento = socket.recv_json()
            fila.put(evento)
    except Exception as erro:
        print(f"[ZMQ] Erro na escuta: {erro}")
    finally:
        socket.close()


def ouvir_camera(previa):
    """Recebe imagens JPEG do emissor, guardando só a mais recente (CONFLATE)."""
    contexto = zmq.Context.instance()
    socket = contexto.socket(zmq.PULL)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.setsockopt(zmq.MAXMSGSIZE, TAMANHO_MAXIMO_IMAGEM)
    try:
        socket.bind(ENDPOINT_CAMERA)
        print(f"[CAMERA] Ouvindo imagens em {ENDPOINT_CAMERA}")
        while True:
            dados = socket.recv()
            imagem = cv2.imdecode(np.frombuffer(dados, dtype=np.uint8), cv2.IMREAD_COLOR)
            if imagem is not None:
                previa.atualizar(imagem)
    except Exception as erro:
        print(f"[CAMERA] Erro na escuta: {erro}")
    finally:
        socket.close()


# =====================================================================
# PROGRAMA PRINCIPAL
# =====================================================================
def montar_tela(estado, previa):
    """Monta o quadro completo: câmera (se ligada) em cima e painel do texto embaixo."""
    painel = np.zeros((ALTURA_PAINEL, LARGURA, 3), dtype=np.uint8)
    desenhar_linhas(painel, montar_linhas(estado.valor))
    desenhar_rodape(painel, estado)
    if not estado.mostrar_camera:
        return painel
    return np.vstack([desenhar_camera(previa.obter(time.monotonic())), painel])


def main():
    fila_eventos = queue.Queue()
    previa = PreviaCamera()
    threading.Thread(target=ouvir_zmq, args=(fila_eventos,), daemon=True).start()
    threading.Thread(target=ouvir_camera, args=(previa,), daemon=True).start()

    estado = Estado()

    print("=== Exibidor de Resultados ===")
    print(f"Modo inicial: {estado.modo} | tecla de reinício: {nome_tecla(estado.tecla_reinicio)}")
    print(f"'{TECLA_MODO}' troca o modo | '{TECLA_TROCAR}' troca a tecla de reinício | "
          f"'{TECLA_LIMPAR}' limpa | '{TECLA_CAMERA}' liga/desliga a câmera | '{TECLA_SAIR}' sai")

    try:
        while True:
            processar_eventos(fila_eventos, estado)
            cv2.imshow("Projeto - Exibidor de Traducao", montar_tela(estado, previa))

            tecla = cv2.waitKey(30) & 0xFF
            if tratar_tecla(estado, tecla, fila_eventos):
                break
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()