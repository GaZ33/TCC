# Comparação: Random Forest, SVM e MLP

Avaliação "pessoa de fora" (3 rodadas). Os hiperparâmetros de cada modelo foram escolhidos só com as pessoas de treino de cada rodada. O comitê faz a média das probabilidades dos três modelos de cada rodada (o tempo dele é a soma dos três).

| Modelo | Acerto por frame | F1 macro | Acerto por tomada | Treino (s) | Classificar 1 frame (ms) |
|---|---:|---:|---:|---:|---:|
| Random Forest | 64,2% | 0,61 | 70,0% | 2,1 | 15,70 |
| SVM | 64,8% | 0,62 | 73,2% | 5,5 | 0,75 |
| MLP | 61,0% | 0,58 | 67,5% | 15,6 | 0,39 |
| Comitê (média dos 3) | 63,6% | 0,61 | 71,0% | 23,1 | 16,84 |

## Por pessoa testada

| Modelo | Pessoa | Acerto por frame | F1 macro | Acerto por tomada |
|---|---|---:|---:|---:|
| Random Forest | Gabriel | 63,7% | 0,59 | 68,0% |
| Random Forest | Fernando | 64,5% | 0,60 | 65,4% |
| Random Forest | Giovanni | 64,5% | 0,63 | 76,5% |
| SVM | Gabriel | 66,9% | 0,65 | 72,0% |
| SVM | Fernando | 65,2% | 0,62 | 73,1% |
| SVM | Giovanni | 62,4% | 0,60 | 74,5% |
| MLP | Gabriel | 58,8% | 0,56 | 62,7% |
| MLP | Fernando | 64,6% | 0,61 | 71,2% |
| MLP | Giovanni | 59,5% | 0,57 | 68,6% |
| Comitê (média dos 3) | Gabriel | 64,4% | 0,61 | 69,3% |
| Comitê (média dos 3) | Fernando | 65,6% | 0,62 | 71,2% |
| Comitê (média dos 3) | Giovanni | 60,9% | 0,59 | 72,5% |

## Hiperparâmetros escolhidos em cada rodada

- Random Forest, teste Gabriel: `{'max_features': 'sqrt', 'min_samples_leaf': 5}` (acerto na validação interna 56,0%)
- SVM, teste Gabriel: `{'C': 0.3, 'gamma': 0.003}` (acerto na validação interna 56,2%)
- MLP, teste Gabriel: `{'hidden_layer_sizes': (512, 256), 'alpha': 0.1}` (acerto na validação interna 52,9%)
- Comitê (média dos 3), teste Gabriel: `os escolhidos acima para cada modelo` (acerto na validação interna nan%)
- Random Forest, teste Fernando: `{'max_features': 0.5, 'min_samples_leaf': 1}` (acerto na validação interna 56,2%)
- SVM, teste Fernando: `{'C': 1, 'gamma': 0.003}` (acerto na validação interna 57,5%)
- MLP, teste Fernando: `{'hidden_layer_sizes': (512, 256), 'alpha': 0.1}` (acerto na validação interna 54,3%)
- Comitê (média dos 3), teste Fernando: `os escolhidos acima para cada modelo` (acerto na validação interna nan%)
- Random Forest, teste Giovanni: `{'max_features': 'sqrt', 'min_samples_leaf': 5}` (acerto na validação interna 51,4%)
- SVM, teste Giovanni: `{'C': 0.1, 'gamma': 0.01}` (acerto na validação interna 57,0%)
- MLP, teste Giovanni: `{'hidden_layer_sizes': (256, 128), 'alpha': 0.0001}` (acerto na validação interna 53,5%)
- Comitê (média dos 3), teste Giovanni: `os escolhidos acima para cada modelo` (acerto na validação interna nan%)

## Acerto por letra (todas as pessoas testadas juntas)

| Letra | Random Forest | SVM | MLP | Comitê (média dos 3) |
|---|---:|---:|---:|---:|
| N | 25% | 54% | 68% | 62% |
| U | 33% | 45% | 50% | 47% |
| M | 33% | 34% | 13% | 27% |
| Q | 35% | 22% | 23% | 25% |
| P | 36% | 57% | 41% | 45% |
| G | 41% | 75% | 57% | 48% |
| X | 43% | 47% | 38% | 40% |
| T | 49% | 0% | 30% | 29% |
| S | 51% | 81% | 38% | 57% |
| K | 55% | 72% | 27% | 44% |
| V | 55% | 47% | 47% | 51% |
| J | 59% | 39% | 50% | 53% |
| Z | 59% | 51% | 55% | 62% |
| F | 64% | 88% | 70% | 65% |
| I | 69% | 86% | 87% | 85% |
| R | 72% | 74% | 47% | 46% |
| E | 82% | 92% | 88% | 91% |
| H | 82% | 51% | 80% | 81% |
| Y | 87% | 88% | 85% | 86% |
| L | 90% | 95% | 80% | 95% |
| W | 91% | 86% | 83% | 86% |
| A | 91% | 82% | 79% | 80% |
| C | 91% | 94% | 90% | 91% |
| D | 94% | 91% | 81% | 89% |
| B | 95% | 97% | 94% | 95% |
| O | 97% | 94% | 97% | 97% |

Figura: `comparacao_modelos.png`.
