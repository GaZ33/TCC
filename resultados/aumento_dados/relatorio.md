# Aumento de dados

Random Forest, avaliação "pessoa de fora". 3 frames artificiais por frame real, gerados só a partir das pessoas de treino. Intensidades: rotação até 15°, proporção dos dedos ±10%, ruído de 1% do tamanho da mão.

| Configuração | Acerto por frame | F1 macro | Acerto por tomada |
|---|---:|---:|---:|
| sem aumento | 63,6% | 0,60 | 71,5% |
| só rotação | 62,2% | 0,59 | 68,4% |
| só proporção dos dedos | 63,6% | 0,60 | 70,4% |
| só ruído | 63,7% | 0,60 | 72,3% |
| as três juntas | 64,0% | 0,60 | 68,4% |

## Por pessoa testada (acerto por frame)

| Configuração | Gabriel | Fernando | Giovanni |
|---|---:|---:|---:|
| sem aumento | 63,1% | 64,8% | 63,1% |
| só rotação | 59,7% | 64,3% | 62,4% |
| só proporção dos dedos | 64,0% | 64,0% | 62,7% |
| só ruído | 63,8% | 62,6% | 64,7% |
| as três juntas | 61,5% | 64,8% | 65,8% |

Figura: `aumento_dados.png`.
