import queue
import threading
import time
import cv2
import numpy as np
import zmq
 
ENDPOINT_EXIBIDOR = "tcp://127.0.0.1:5557"
 
# =====================================================================
# CONFIGURAÇÕES (ajuste aqui)
# =====================================================================
# Modos de funcionamento (o usuário troca com a tecla TECLA_MODO, com o programa rodando)
MODO_MANUAL = "MANUAL"            # congela no 1º resultado até apertar a tecla de reinício
MODO_AUTOMATICO = "AUTOMATICO"    # atualiza a cada resultado e limpa sozinho (CLEAR do filtro)
 
MODO_PADRAO = MODO_MANUAL
TECLA_REINICIO_PADRAO = " "       # " " = espaço | "\r" = Enter | ou qualquer tecla comum (ex.: "r")
 
# Teclas fixas (não podem ser escolhidas como tecla de reinício)
TECLA_SAIR = "q"
TECLA_MODO = "m"                  # alterna entre MANUAL e AUTOMATICO
TECLA_TROCAR = "t"                # a próxima tecla apertada vira a nova tecla de reinício
TECLA_LIMPAR = "c"                # sempre limpa a tela, nos dois modos
TECLA_ESC = 27                    # cancela a troca de tecla
TECLAS_RESERVADAS = {ord(TECLA_SAIR), ord(TECLA_MODO), ord(TECLA_TROCAR), ord(TECLA_LIMPAR), TECLA_ESC}
 
# --- Aparência do texto na tela ---
FONTE = cv2.FONT_HERSHEY_SIMPLEX
COR_TEXTO = (0, 255, 0)
COR_DICA = (160, 160, 160)
ESCALA_MAX = 2.2     # tamanho usado quando o texto cabe sem problema
ESCALA_RODAPE = 0.5
MARGEM_X = 30        # folga lateral: se o texto passar disso, ele encolhe
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
    AUTOMATICO: cada SHOW atualiza a tela e o CLEAR apaga, como no início."""
    while True:
        try:
            evento = fila.get_nowait()
        except queue.Empty:
            break
 
        if estado.travado:
            continue
 
        acao = evento.get("acao")
        valor = evento.get("valor")
 
        if acao == "SHOW" and isinstance(valor, str) and valor:
            estado.valor = valor.upper()
            if estado.modo == MODO_MANUAL:
                estado.travado = True
        elif acao == "CLEAR":
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
        linha1 = "Modo AUTOMATICO: atualiza e limpa sozinho"
    elif estado.travado:
        linha1 = f"CONGELADO: aperte {nome} para reiniciar"
    else:
        linha1 = f"Modo MANUAL: {nome} reinicia depois do resultado"
    linha2 = f"{TECLA_MODO}: trocar modo | {TECLA_TROCAR}: trocar tecla | {TECLA_SAIR}: sair"
    return linha1, linha2
 
 
def desenhar_rodape(frame, estado):
    altura = frame.shape[0]
    linha1, linha2 = textos_rodape(estado, time.monotonic())
    cv2.putText(frame, linha1, (MARGEM_X, altura - 28), FONTE, ESCALA_RODAPE, COR_DICA, 1, cv2.LINE_AA)
    cv2.putText(frame, linha2, (MARGEM_X, altura - 10), FONTE, ESCALA_RODAPE, COR_DICA, 1, cv2.LINE_AA)
 
 
# =====================================================================
# ZEROMQ E PROGRAMA PRINCIPAL
# =====================================================================
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
 
def main():
    fila_eventos = queue.Queue()
    thread_recepcao = threading.Thread(target=ouvir_zmq, args=(fila_eventos,), daemon=True)
    thread_recepcao.start()
 
    # Cria uma janela/fundo estático (sem usar a webcam)
    largura, altura = 640, 200
    estado = Estado()
 
    print("=== Exibidor de Resultados (Sem Webcam) ===")
    print(f"Modo inicial: {estado.modo} | tecla de reinício: {nome_tecla(estado.tecla_reinicio)}")
    print(f"'{TECLA_MODO}' troca o modo | '{TECLA_TROCAR}' troca a tecla de reinício | "
          f"'{TECLA_LIMPAR}' limpa | '{TECLA_SAIR}' sai")
 
    try:
        while True:
            # Cria um fundo preto fixo para mostrar a letra traduzida
            frame = np.zeros((altura, largura, 3), dtype=np.uint8)
 
            processar_eventos(fila_eventos, estado)
 
            # Escreve na tela: "Letra: X", "Numero: 5" ou o aviso de O/0
            desenhar_linhas(frame, montar_linhas(estado.valor))
            desenhar_rodape(frame, estado)
 
            cv2.imshow("Projeto - Exibidor de Traducao", frame)
 
            tecla = cv2.waitKey(30) & 0xFF
            if tratar_tecla(estado, tecla, fila_eventos):
                break
 
    finally:
        cv2.destroyAllWindows()
 
if __name__ == "__main__":
    main()
 