import os
import zmq
import pandas as pd

# =====================================================================
# CONFIGURAÇÕES
# =====================================================================
ENDERECO = "tcp://127.0.0.1:5555"      # o receptor faz bind; o emissor faz connect no mesmo endereço
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PASTA_SAIDA = os.path.join(SCRIPT_DIR, "saida_csv")


def criar_arquivo_novo(pasta, nome_base, extensao=".csv"):
    """Cria e abre um arquivo que ainda NÃO existe.
    Tenta 'nome.csv'; se já existir, tenta 'nome_1.csv', 'nome_2.csv'... e assim por diante.
    O modo 'x' só abre se o arquivo não existir (se existir, dá FileExistsError),
    então nunca há risco de sobrescrever nada.
    Retorna (arquivo_aberto, caminho)."""
    n = 0
    while True:
        nome = f"{nome_base}{extensao}" if n == 0 else f"{nome_base}_{n}{extensao}"
        caminho = os.path.join(pasta, nome)
        try:
            arquivo = open(caminho, "x", newline="", encoding="utf-8")
            return arquivo, caminho
        except FileExistsError:
            n += 1


os.makedirs(PASTA_SAIDA, exist_ok=True)

contexto = zmq.Context()
socket_recepcao = contexto.socket(zmq.PULL)
socket_recepcao.bind(ENDERECO)

print(f"Aguardando janelas em {ENDERECO} ...")
print(f"Os CSVs serão salvos em: {PASTA_SAIDA}")
print("Pressione Ctrl+C para encerrar.")

try:
    while True:
        # Espera até 500 ms por mensagem. Fica em loop para o Ctrl+C funcionar no Windows.
        if not socket_recepcao.poll(500):
            continue

        # ATENÇÃO: recv_pyobj usa pickle. Só receba de um emissor em que você confia
        # (por isso o endereço acima é local: 127.0.0.1).
        pacote = socket_recepcao.recv_pyobj()

        try:
            sequencia = pacote["sequencia"]
            colunas = pacote["colunas"]
            amostras = pacote["amostras"]
            step = pacote.get("step")
            regime = pacote.get("regime")
            com_mao = pacote.get("frames_com_mao")
        except (KeyError, TypeError, AttributeError):
            print("[AVISO] Mensagem em formato inesperado, ignorada.")
            continue

        df = pd.DataFrame(amostras, columns=colunas)

        arquivo, caminho = criar_arquivo_novo(PASTA_SAIDA, f"janela_{sequencia:04d}")
        with arquivo:
            df.to_csv(arquivo, index=False)

        print(f"[RECEBIDO] janela {sequencia} | {len(df)} amostras | step={step} | regime={regime} | com mão={com_mao} -> {os.path.basename(caminho)}")

except KeyboardInterrupt:
    print("\nEncerrado pelo usuário.")
finally:
    socket_recepcao.close()
    contexto.term()
