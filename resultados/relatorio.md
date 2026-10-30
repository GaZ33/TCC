# Resultados do classificador de letras

Gerado por `treinar_classificador.py` em 30/10/2026 16:44. Random Forest com 300 árvores (semente 42), scikit-learn 1.9.1, classes: letras (26).

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
| Gabriel | 6716 | 75 | 63,1% | 0,58 | 70,7% |
| Fernando | 3315 | 52 | 64,8% | 0,59 | 65,4% |
| Giovanni | 4497 | 51 | 63,1% | 0,62 | 78,4% |
| **Média** | 14528 | 178 | 63,6% | 0,60 | 71,5% |

## Acerto por letra

Da pior para a melhor (todas as pessoas testadas juntas): N 19%, S 33%, M 34%, U 35%, P 38%, Q 40%, G 41%, X 41%, K 52%, F 52%, V 52%, J 61%, Z 61%, T 62%, R 67%, I 69%, E 82%, W 86%, Y 86%, H 87%, A 91%, C 91%, D 92%, L 92%, B 95%, O 97%.

Figuras: `acerto_por_classe.png` e `matriz_confusao.png`.

## Confusões mais comuns

| Letra real | Prevista como | Frames |
|---|---|---:|
| K | H | 289 |
| J | I | 276 |
| U | R | 255 |
| M | W | 249 |
| G | Z | 222 |
| T | F | 218 |
| F | T | 214 |
| S | I | 191 |
| Q | Z | 178 |
| V | H | 147 |

## Comparação: frames sorteados (ilusório)

Sorteando 25% dos frames para teste, o acerto é 98,9%. O número é ilusório: frames da mesma tomada caem no treino e no teste. Não usar como resultado.

## Modelo final

Treinado com todos os 14528 frames das 3 pessoas e salvo em `modelos/random_forest_letras.joblib`.
