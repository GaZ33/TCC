# TCC

Reconhecimento do alfabeto manual de Libras com landmarks da mão (MediaPipe) e Random Forest.

## Instalação (uma vez)

No PowerShell, dentro da pasta do projeto:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install mediapipe==1.1.0 pandas==3.0.6 scikit-learn
```

O mediapipe já instala o OpenCV (`opencv-contrib-python`). Não instale também o `opencv-python`: com os dois juntos, o `cv2` quebra.

## 1. Gravar dados

```powershell
.venv\Scripts\python novos_codigos\coleta_hand_landmarks_com_label.py
```

Clique na janela da câmera. Aperte uma letra para escolher o que será gravado, TAB para escolher a mão (direita/esquerda) e ESPAÇO para gravar 90 frames. ESC sai. Cada tomada vira `letra <x> <mão>.csv` e `.mp4` (nunca sobrescreve: repetições ganham `_2`, `_3`...). Se a tela avisar "Mao detectada: ... (DIFERENTE!)", a mão na tela não é a escolhida no TAB.

## 2. Treinar

```powershell
.venv\Scripts\python treinar_classificador.py
```

Cada pasta em `DATASETS` (no topo do arquivo) é o dataset de uma pessoa. O script treina com todas menos uma e testa na que ficou de fora, para cada pessoa. **Esse acerto "pessoa de fora" é o número do TCC**: mede o modelo com alguém que ele nunca viu. O acerto com frames sorteados (~99%) aparece só para comparação e é ilusório. No fim, o modelo treinado com todas as pessoas é salvo em `modelos/random_forest_letras.joblib`.

## 3. Classificar ao vivo

```powershell
.venv\Scripts\python classificar_ao_vivo.py
.venv\Scripts\python classificar_ao_vivo.py --video "novos_codigos\letra a direita.mp4"
```

Mostra a letra mais votada nos últimos 10 frames com mão (precisa de 6 votos; senão mostra "?") e as 3 letras mais prováveis do frame atual. A mão é detectada sozinha; TAB alterna entre automático, direita fixa e esquerda fixa. ESC sai. Com `--video`, processa um vídeo do coletor e imprime a letra mais mostrada.

O modelo foi treinado com quem gravou os dados, então ao vivo com essas pessoas ele acerta mais do que com uma pessoa nova. Para o acerto real, use a avaliação "pessoa de fora" do treino.

## Testes

```powershell
.venv\Scripts\python -m unittest discover -s testes
```
