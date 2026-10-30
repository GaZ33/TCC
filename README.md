# TCC

Reconhecimento do alfabeto manual de Libras com landmarks da mão (MediaPipe) e Random Forest, em tempo real.

```
emissor_janela_zmq_sem_tela ──5555──> classificador ──5556──> filtro ──5557──> exibidor_texto
             └───────────────────── 5558 (imagem da câmera) ─────────────────────┘
```

| Arquivo | O que faz |
|---|---|
| `emissor_janela_zmq_sem_tela.py` | Abre a câmera, extrai os 21 landmarks da mão e envia a janela deslizante de frames ao classificador e a imagem ao exibidor |
| `classificador.py` | Calcula as características de cada frame novo e as probabilidades de cada letra (Random Forest) |
| `filtro.py` | Escolhe a letra mais provável e só a repassa se a confiança passar do limiar |
| `exibidor_texto.py` | Mostra a imagem da câmera e a letra reconhecida |
| `caracteristicas_mao.py` | As 35 características da mão (ângulos, distâncias, orientação), usadas no treino e no classificador |
| `treinar_classificador.py` | Treina e avalia o modelo; gera `modelos/` e `resultados/` |
| `comparar_modelos.py` / `testar_aumento_dados.py` | Experimentos: comparação de classificadores e teste do aumento de dados (`aumento_dados.py`) |
| `novos_codigos/coleta_hand_landmarks_com_label.py` | Coletor dos datasets (uma tomada de 90 frames por letra) |

## Instalação (uma vez)

No PowerShell, dentro da pasta do projeto:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

## 1. Gravar dados

```powershell
.venv\Scripts\python novos_codigos\coleta_hand_landmarks_com_label.py
```

Clique na janela da câmera. Aperte uma letra para escolher o que será gravado, TAB para escolher a mão (direita/esquerda) e ESPAÇO para gravar 90 frames. ESC sai. Cada tomada vira `letra <x> <mão>.csv` e `.mp4` (nunca sobrescreve: repetições ganham `_2`, `_3`...). Se a tela avisar "Mao detectada: ... (DIFERENTE!)", a mão na tela não é a escolhida no TAB.

## 2. Treinar

```powershell
.venv\Scripts\python treinar_classificador.py
```

Cada pasta em `DATASETS` (no topo do arquivo) é o dataset de uma pessoa. O script treina com todas menos uma e testa na que ficou de fora, para cada pessoa. **Esse acerto "pessoa de fora" é o número do TCC**: mede o modelo com alguém que ele nunca viu. No fim, salva:

- `modelos/random_forest_letras.joblib`: o modelo treinado com todas as pessoas (usado pelo `classificador.py`);
- `resultados/`: `relatorio.md`, tabelas CSV (acerto por pessoa, por letra, matriz de confusão) e as figuras `matriz_confusao.png` e `acerto_por_classe.png`.

Experimentos (mesma avaliação "pessoa de fora"; levam alguns minutos):

```powershell
.venv\Scripts\python comparar_modelos.py       # Random Forest x SVM x MLP x comitê -> resultados/comparacao/
.venv\Scripts\python testar_aumento_dados.py   # com e sem aumento de dados (aumento_dados.py) -> resultados/aumento_dados/
```

## 3. Rodar o sistema

Abra quatro terminais na pasta do projeto, nesta ordem (quem recebe começa antes de quem envia):

```powershell
.venv\Scripts\python exibidor_texto.py
.venv\Scripts\python filtro.py
.venv\Scripts\python classificador.py
.venv\Scripts\python emissor_janela_zmq_sem_tela.py
```

Só o emissor usa a câmera. No exibidor: M troca o modo (manual/automático), ESPAÇO limpa, V liga/desliga a imagem, Q sai. Os outros terminais encerram com Ctrl+C.

## Testes

```powershell
.venv\Scripts\python -m unittest discover -s testes
```
