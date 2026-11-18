# Deep Learning para Segmentação de Tumores Cerebrais: Aplicação de U-Net no Dataset LGG-MRI

[![Paper PDF](https://img.shields.io/badge/Artigo-PDF-red.svg)](artigo_segmentacao_lgg_mri.pdf)
[![LaTeX](https://img.shields.io/badge/LaTeX-IEEEtran-blue.svg)](latex/artigo.tex)
[![Dataset](https://img.shields.io/badge/Dataset-TCGA--LGG--MRI-success.svg)](https://wiki.cancerimagingarchive.net/)
[![Instituição](https://img.shields.io/badge/UnB-FCTE-darkblue.svg)](https://fcte.unb.br/)

**Autores:**  
- José Joaquim da Silva Neto (netojoaquimjj@gmail.com | Matrícula: 232027510)  
- Pedro Henrique Gomes (pedrohg2609@gmail.com | Matrícula: 232030041)  

*Faculdade de Ciências e Tecnologias em Engenharia — Universidade de Brasília (UnB), Brasil*

---

## Sumário Executivo

A segmentação automática de gliomas de baixo grau (LGG - *Lower-Grade Gliomas*) em imagens de ressonância magnética (MRI, com foco na sequência FLAIR) é essencial para suporte ao diagnóstico, biópsias estereotáxicas e planejamento cirúrgico e radioterápico em neuro-oncologia. No entanto, o problema impõe severos desafios metodológicos: heterogeneidade textural dos gliomas, variações nos protocolos de aquisição entre instituições hospitalares e desbalanceamento espacial extremo entre o volume tumoral e o tecido cerebral saudável.

Este estudo implementa, avalia e compara três arquiteturas de aprendizado profundo baseadas em U-Net sobre o conjunto público TCGA LGG-MRI (110 pacientes, 5 instituições). O trabalho adota:
1. **Particionamento estrito paciente-disjunto (70/15/15)** estratificado por instituição de aquisição, eliminando vazamento de dados (*data leakage*) entre fatias 2D adjacentes;
2. **Calibração cega de limiar:** otimização do limiar (*threshold*) de binarização conduzida unicamente sobre a validação, avaliando o conjunto de teste de forma estritamente cega;
3. **Desagregação de métricas:** demonstração de que métricas agregadas globais (Dice geral) mascaram o desempenho na região clínica de interesse;
4. **Análise de equidade (*fairness*) com intervalos de confiança via Bootstrap (95%):** investigação de disparidades por subgrupo genômico (*RNASeqCluster*) e por grau histológico (*Neoplasm Histologic Grade*).

---

## Protocolo Experimental e Metodologia

### 1. Base de Dados e Divisão dos Pacientes
- **Dataset:** LGG-MRI Segmentation (The Cancer Genome Atlas - TCGA / The Cancer Imaging Archive - TCIA).
- **Amostra Total:** 110 pacientes, exames com fatias axiais 2D de sequências FLAIR e máscaras binárias segmentadas manualmente por especialistas.
- **Divisão Paciente-Disjunta (70% Treino / 15% Validação / 15% Teste):**
  - **Treino:** 76 pacientes (2.809 fatias 2D);
  - **Validação:** 17 pacientes (565 fatias 2D) — utilizado para calibrar hiperparâmetros e limiares de decisão;
  - **Teste:** 17 pacientes (555 fatias 2D) — conjunto intocado e avaliado apenas na etapa final.
- **Estratificação:** O particionamento preservou a proporção de pacientes provenientes de cada uma das 5 instituições de saúde de origem.

### 2. Arquiteturas Avaliadas
- **U-Net Baseline:** Arquitetura codificador-decodificador com conexões de salto diretas (*skip connections*) e *backbone* ResNet-34 pré-treinado na ImageNet.
- **Attention U-Net:** Adição de *Attention Gates* (AGs) nas conexões de salto antes da concatenação. O módulo de atenção suprime ativações espaciais irrelevantes no tecido saudável e realça características compatíveis com a escala tumoral.
- **Attention U-Net + Loss Tversky:** Combinação da Attention U-Net com a função de perda focal de Tversky ($\alpha=0{,}3, \beta=0{,}7$) associada a amostragem balanceada via `WeightedRandomSampler` para mitigar o desbalanceamento entre fatias sem tumor e com tumor.

### 3. Métricas de Avaliação
O desempenho pixel a pixel foi mensurado com base na matriz de confusão (Verdadeiros Positivos - $VP$, Falsos Positivos - $FP$, Falsos Negativos - $FN$):

$$\text{Dice} = \frac{2 |P \cap G|}{|P| + |G|} = \frac{2 \cdot VP}{2 \cdot VP + FP + FN}$$

$$\text{IoU (Jaccard)} = \frac{|P \cap G|}{|P \cup G|} = \frac{VP}{VP + FP + FN}$$

- **Dice Geral:** Macro-média aritmética calculada fatia a fatia.
- **IoU Agregado:** Micro-média calculada acumulando a totalidade de pixels do conjunto de teste (pondera naturalmente regiões tumorais de maior extensão).
- **Incerteza Estatística:** Intervalos de confiança de 95% (IC 95%) calculados por *bootstrap* estratificado a nível de paciente com 500 reamostragens.

---

## Resultados Experimentais

### Desempenho no Conjunto de Teste (17 Pacientes, 555 Fatias)

| Modelo | Limiar Ótimo (Val) | Dice Geral | Dice (c/ Tumor) | Dice (s/ Tumor) | Precisão | Recall (Sensibilidade) | IoU Agregado |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **U-Net (Baseline)** | 0,85 | **0,8520** | **0,7401** | **0,9116** | **0,8877** | 0,7447 | 0,6805 |
| **Attention U-Net** | 0,60 | 0,8364 | 0,7161 | 0,9006 | 0,8276 | 0,7957 | 0,6826 |
| **Attention U-Net + Tversky** | 0,85 | 0,8300 | 0,7340 | 0,8812 | 0,8322 | **0,8116** | **0,6974** |

---

## Principais Descobertas Científicas

### 1. O Paradoxo das Métricas Agregadas
Métricas agregadas globais frequentemente reportadas na literatura (Dice geral em torno de 0,85) não refletem a acurácia estrita na região clínica relevante:
- Nas fatias **sem lesão tumoral**, o Dice atinge valores entre **0,8812 e 0,9116**;
- Nas fatias **com presença de tumor**, o Dice declina para a faixa de **0,7161 a 0,7401** (uma diferença de 15 a 19 pontos percentuais).
Métricas agregadas, portanto, superestimam a eficácia da delimitação volumétrica tumoral ao serem infladas pelas predições de fatias negativas.

### 2. O Trade-Off Clínico: Precisão vs. Recall
À medida que se introduzem *Attention Gates* e a função de perda Tversky orientada a recall ($\beta=0{,}7$):
- O **Recall (sensibilidade)** sobe sistematicamente de **0,7447** (Baseline) para **0,7957** (Attention) e **0,8116** (Tversky);
- A **Precisão** declina de **0,8877** para **0,8276** e **0,8322**;
- O **IoU Agregado** atinge o valor mais alto na variante Tversky (**0,6974**), demonstrando melhor captura em lesões de maior extensão volumétrica.
Em cenários de triagem médica onde o custo de falsos negativos é prioritário, a variante **Tversky** apresenta o perfil mais balanceado para intervenção precoce.

---

## Análise de Equidade e Fairness

### A. Subgrupo Genômico (*RNASeqCluster*)
Distribuição de desempenho entre os clusters moleculares:

| Subgrupo Genômico | Pacientes (Teste) | Dice U-Net Baseline | IC 95% U-Net | Dice Attention U-Net | Dice Attention + Tversky |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Cluster 1** | 2 | 0,9267 | [0,8977; 0,9432] | 0,8785 | 0,9063 |
| **Cluster 2** | 2 | 0,8953 | [0,8845; 0,9024] | 0,8797 | 0,8591 |
| **Cluster 3** | 3 | 0,8303 | [0,7791; 0,9095] | 0,8449 | 0,8336 |
| **Cluster 4** | 5 | 0,9225 | [0,9024; 0,9420] | 0,9039 | 0,8981 |

Apesar de o Cluster 3 apresentar média inferior, os intervalos de confiança com 95% de significância sobrepõem-se expressivamente aos demais grupos. Com 2 a 5 pacientes por partição de teste, a incerteza amostral impede afirmações estatisticamente conclusivas de viés genômico.

### B. Grau Histológico do Tumor (*Neoplasm Histologic Grade*)
Comportamento dos modelos entre graus de malignidade tecidual:

| Grau Histológico | Pacientes (Teste) | U-Net Baseline (IC 95%) | Attention U-Net (IC 95%) | Attention + Tversky (IC 95%) |
| :--- | :---: | :---: | :---: | :---: |
| **Grau 2** | 10 | **0,8820** [0,8498; 0,9105] | **0,8792** [0,8545; 0,9004] | **0,8703** [0,8369; 0,8957] |
| **Grau 1** | 7 | **0,8101** [0,7552; 0,8975] | **0,7768** [0,7215; 0,8562] | **0,7740** [0,6967; 0,8806] |

A estratificação por grau histológico revelou-se consistente e estatisticamente defensável: tumores de **Grau 1** apresentaram coeficientes Dice sistematicamente inferiores (de 7 a 10 pontos percentuais a menos) aos de Grau 2 em todos os três modelos avaliados, refletindo o desafio de demarcação de bordas menos contrastadas típicas de lesões de menor grau.

---

## Inspeção Qualitativa

O arquivo visual de segmentação comparativa está localizado em:
`latex/exemplo_segmentacao.png`

O arquivo exibe a imagem FLAIR original, a máscara de referência manual e as segmentações inferidas pelos três modelos analisados.

---

## Estrutura de Diretórios e Compilação

```
01-segmentacao-tumores-cerebrais-unet/
├── README.md
├── artigo_segmentacao_lgg_mri.pdf
└── latex/
    ├── artigo.tex
    ├── IEEEtran.cls
    └── exemplo_segmentacao.png
```

### Compilação do Artigo via TeX Live / pdflatex
```bash
cd latex
pdflatex artigo.tex
pdflatex artigo.tex
```

---

## Referência Bibliográfica

```bibtex
@inproceedings{silvaneto2026segmentacao,
  author    = {Silva Neto, Jos{\'e} Joaquim da and Gomes, Pedro Henrique},
  title     = {Deep Learning para Segmenta{\c{c}}{\~a}o de Tumores Cerebrais: Aplica{\c{c}}{\~a}o de U-Net no Dataset LGG-MRI},
  booktitle = {Anais da Disciplina de Intelig{\^e}ncia Artificial 2 - UnB FCTE},
  year      = {2026},
  address   = {Bras{\'i}lia, Brasil}
}
```
