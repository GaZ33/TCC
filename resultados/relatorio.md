# Resultados do classificador de letras

Gerado por `treinar_classificador.py` em 30/10/2026 18:06. Modelo: SVM com kernel RBF (C = 0,3, gamma = 0,003), probabilidades calibradas; scikit-learn 1.9.1; classes: letras (26).

## Dados

| Pessoa | Tomadas | Frames com mão | Frames sem mão (descartados) |
|---|---:|---:|---:|
| Gabriel | 98 | 8772 | 48 |
| Fernando | 68 | 4164 | 1956 |
| Giovanni | 69 | 6117 | 93 |

## Avaliação "pessoa de fora"

Para cada pessoa, o modelo é treinado com as outras e testado nela. O acerto por tomada é a letra mais prevista entre os frames da tomada.

| Pessoa testada | Frames | Tomadas | Acerto por frame | F1 macro | Acerto por tomada |
|---|---:|---:|---:|---:|---:|
| Gabriel | 6716 | 75 | 66,1% | 0,64 | 72,0% |
| Fernando | 3315 | 52 | 69,6% | 0,66 | 78,8% |
| Giovanni | 4497 | 51 | 63,0% | 0,60 | 76,5% |
| **Média** | 14528 | 178 | 66,2% | 0,63 | 75,8% |

## Acerto por letra

Da pior para a melhor (todas as pessoas testadas juntas): T 0%, J 26%, Q 27%, M 38%, U 48%, Z 48%, H 50%, N 52%, X 53%, P 58%, V 59%, G 69%, R 71%, K 72%, A 84%, Y 88%, S 88%, W 89%, D 89%, E 92%, I 94%, C 95%, L 95%, O 96%, F 96%, B 96%.

Figuras: `acerto_por_classe.png` e `matriz_confusao.png`.

## Confusões mais comuns

| Letra real | Prevista como | Frames |
|---|---|---:|
| T | F | 583 |
| J | I | 409 |
| H | K | 245 |
| Q | Z | 216 |
| P | K | 194 |
| U | R | 179 |
| V | R | 171 |
| M | N | 160 |
| N | M | 151 |
| Z | G | 130 |

## Comparação: frames sorteados (ilusório)

Sorteando 25% dos frames para teste, o acerto é 84,0%. O número é ilusório: frames da mesma tomada caem no treino e no teste. Não usar como resultado.

## Modelo final

Treinado com todos os 14528 frames das 3 pessoas e salvo em `modelos/classificador_letras.joblib`. Tempo para classificar um quadro: 7,06 ms.
