# Deep Learning para Segmentação de Tumores Cerebrais: Aplicação de U-Net no Dataset LGG-MRI

[![Paper PDF](https://img.shields.io/badge/Artigo-PDF-red.svg)](artigo_segmentacao_lgg_mri.pdf)
[![LaTeX](https://img.shields.io/badge/LaTeX-IEEEtran-blue.svg)](latex/artigo.tex)
[![Dataset](https://img.shields.io/badge/Dataset-TCGA--LGG--MRI-success.svg)](https://wiki.cancerimagingarchive.net/)
[![Instituição](https://img.shields.io/badge/UnB-FCTE-darkblue.svg)](https://fcte.unb.br/)

> **Autores:**  
> - **José Joaquim da Silva Neto** (netojoaquimjj@gmail.com | Matrícula: 232027510)  
> - **Pedro Henrique Gomes** (pedrohg2609@gmail.com | Matrícula: 232030041)  
>  
> *Faculdade de Ciências e Tecnologias em Engenharia — Universidade de Brasília (UnB), Brasil*

---

## 📌 Sumário Executivo

A segmentação automática de gliomas de baixo grau (**LGG** - *Lower-Grade Gliomas*) em imagens de ressonância magnética (**MRI**, com foco na sequência **FLAIR**) é essencial para o suporte ao diagnóstico, biópsias estereotáxicas e planejamento cirúrgico e radioterápico em neuro-oncologia. No entanto, o problema impõe severos desafios metodológicos: forte heterogeneidade textural dos gliomas, variações nos protocolos de aquisição entre instituições hospitalares e desbalanceamento espacial extremo entre o volume tumoral e o tecido cerebral saudável.

Este estudo implementa, avalia e compara rigorosamente três arquiteturas de aprendizado profundo baseadas em U-Net sobre o conjunto público **TCGA LGG-MRI** (110 pacientes, 5 instituições). O trabalho se diferencia na literatura ao adotar:
1. **Particionamento estrito paciente-disjunto (70/15/15)** estratificado por instituição de aquisição, eliminando vazamento de dados (*data leakage*) entre fatias 2D adjacentes;
2. **Calibração cega de limiar:** otimização do *threshold* de binarização conduzida unicamente sobre a validação, avaliando o conjunto de teste de forma estritamente cega;
3. **Desagregação de métricas:** revelação de que métricas agregadas globais (Dice geral) mascaram o desempenho na região clínica de interesse;
4. **Análise de Equidade (*Fairness*) com Intervalos de Confiança via Bootstrap (95%):** investigação de disparidades por subgrupo genômico (*RNASeqCluster*) e por grau histológico (*Neoplasm Histologic Grade*).

---

## 🔬 Protocolo Experimental e Metodologia

### 1. Base de Dados e Divisão dos Pacientes
- **Dataset:** LGG-MRI Segmentation (The Cancer Genome Atlas - TCGA / The Cancer Imaging Archive - TCIA).
- **Amostra Total:** 110 pacientes, exames com fatias axiais 2D de sequências FLAIR e máscaras binárias segmentadas manualmente por especialistas.
- **Divisão Paciente-Disjunta (70% Treino / 15% Validação / 15% Teste):**
  - **Treino:** 76 pacientes (2.809 fatias 2D);
  - **Validação:** 17 pacientes (565 fatias 2D) — usado para calibrar hiperparâmetros e limiares de decisão;
  - **Teste:** 17 pacientes (555 fatias 2D) — conjunto intocado e avaliado apenas ao final.
- **Estratificação:** O *split* preservou a proporção de pacientes provenientes de cada uma das 5 instituições de saúde de origem.

### 2. Arquiteturas Avaliadas
- **U-Net Baseline:** Arquitetura codificador-decodificador com conexões de salto (*skip connections* diretas) e *backbone* **ResNet-34** pré-treinado no ImageNet.
- **Attention U-Net:** Adição de *Attention Gates* (AGs) nas conexões de salto antes da concatenação. O módulo de atenção suprime ativações espaciais irrelevantes no tecido saudável de fundo e realça sinais salientes compatíveis com a escala tumoral.
- **Attention U-Net + Loss Tversky:** Combinação do modelo com Attention Gates à função de perda focal de **Tversky** ($\alpha=0{,}3, \beta=0{,}7$) associada a amostragem balanceada via `WeightedRandomSampler` para combater o desbalanceamento entre fatias sem tumor e com tumor.

### 3. Equações das Métricas de Avaliação
O desempenho pixel a pixel foi mensurado com base na matriz de confusão (Verdadeiros Positivos - $VP$, Falsos Positivos - $FP$, Falsos Negativos - $FN$):

$$\text{Dice} = \frac{2 |P \cap G|}{|P| + |G|} = \frac{2 \cdot VP}{2 \cdot VP + FP + FN}$$

$$\text{IoU (Jaccard)} = \frac{|P \cap G|}{|P \cup G|} = \frac{VP}{VP + FP + FN}$$

- **Dice Geral:** Macro-média aritmética calculada fatia a fatia.
- **IoU Agregado:** Micro-média calculada acumulando a totalidade de pixels do conjunto de teste (pondera naturalmente regiões tumorais de maior extensão).
- **Incerteza Estatística:** Intervalos de confiança de 95% (IC 95%) calculados por *bootstrap* estratificado a nível de paciente com 500 reamostragens.

---

## 📊 Resultados Experimentais

### Desempenho no Conjunto de Teste (17 Pacientes, 555 Fatias)

| Modelo | Threshold Ótimo (Val) | Dice Geral | Dice (c/ Tumor) | Dice (s/ Tumor) | Precisão | Recall (Sensib.) | IoU Agregado |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **U-Net (Baseline)** | 0,85 | **0,8520** | **0,7401** | **0,9116** | **0,8877** | 0,7447 | 0,6805 |
| **Attention U-Net** | 0,60 | 0,8364 | 0,7161 | 0,9006 | 0,8276 | 0,7957 | 0,6826 |
| **Attention U-Net + Tversky** | 0,85 | 0,8300 | 0,7340 | 0,8812 | 0,8322 | **0,8116** | **0,6974** |

---

## 💡 Principais Descobertas Científicas

### 1. O Paradoxo das Métricas Agregadas
Muitos artigos da área reportam apenas o Dice geral agregado (~0,85), sugerindo excelente acurácia. Contudo, ao estratificar a avaliação:
- Nas fatias **sem qualquer lesão**, o Dice fica entre **0,8812 e 0,9116**;
- Nas fatias **efetivamente com tumor**, o Dice cai substancialmente para a faixa de **0,7161 a 0,7401** (uma defasagem de 15 a 19 pontos percentuais).
*Conclusão clínica:* Métricas agregadas globais mascaram a verdadeira dificuldade do modelo em delimitar com precisão as margens tumorais nas fatias patológicas.

### 2. O Trade-Off Clínico: Precisão vs. Recall
À medida que se introduzem Attention Gates e a função de perda Tversky orientada a falsos negativos ($\beta=0{,}7$):
- O **Recall (sensibilidade)** sobe sistematicamente de **0,7447** (Baseline) $\rightarrow$ **0,7957** (Attention) $\rightarrow$ **0,8116** (Tversky);
- A **Precisão** cai de **0,8877** $\rightarrow$ **0,8276** / **0,8322**;
- O **IoU Agregado** atinge seu ápice na variante Tversky (**0,6974**), demonstrando que ela delimita muito melhor tumores de grande volume anatômico.
*Aplicação prática:* Em triagem clínica neuro-oncológica, onde o custo de omitir tecido tumoral (falso negativo) é crítico, a variante **Tversky** apresenta o perfil mais desejável, apesar de penalizar levemente o Dice médio por fatia.

---

## ⚖️ Análise de Equidade e Fairness

Para examinar se os modelos apresentam comportamentos tendenciosos em diferentes subgrupos de pacientes, foram avaliadas duas variáveis clínicas de metadados:

### A. Subgrupo Genômico (*RNASeqCluster*)
Os pacientes são divididos em 4 clusters de expressão de RNA:

| Subgrupo Genômico | Nº Pacientes (Teste) | Dice U-Net Baseline | IC 95% U-Net | Dice Attention U-Net | Dice Attention + Tversky |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Cluster 1** | 2 | 0,9267 | [0,8977; 0,9432] | 0,8785 | 0,9063 |
| **Cluster 2** | 2 | 0,8953 | [0,8845; 0,9024] | 0,8797 | 0,8591 |
| **Cluster 3** | 3 | 0,8303 | [0,7791; 0,9095] | 0,8449 | 0,8336 |
| **Cluster 4** | 5 | 0,9225 | [0,9024; 0,9420] | 0,9039 | 0,8981 |

> **Achado Metodológico:** Embora o Cluster 3 apresente consistentemente a menor média de Dice (0,83 a 0,84), os intervalos de confiança via bootstrap se sobrepõem expressivamente. Com apenas 2 a 5 pacientes por grupo no teste, a **alta incerteza amostral** impede que essa diferença seja afirmada como um viés estatisticamente conclusivo.

### B. Grau Histológico do Tumor (*Neoplasm Histologic Grade*)
Classificação da agressividade tecidual tumoral (Grau 1 vs. Grau 2):

| Grau Histológico | Nº Pacientes (Teste) | U-Net Baseline (IC 95%) | Attention U-Net (IC 95%) | Attention + Tversky (IC 95%) |
| :--- | :---: | :---: | :---: | :---: |
| **Grau 2** | 10 | **0,8820** [0,8498; 0,9105] | **0,8792** [0,8545; 0,9004] | **0,8703** [0,8369; 0,8957] |
| **Grau 1** | 7 | **0,8101** [0,7552; 0,8975] | **0,7768** [0,7215; 0,8562] | **0,7740** [0,6967; 0,8806] |

> **Achado Defensável e Robusto:** A análise por grau histológico demonstrou-se robusta. Em todos os três modelos sem exceção, os tumores de **Grau 1** apresentaram um Dice sistematicamente inferior (cerca de **7 a 10 pontos percentuais a menos**) que os de Grau 2. Isso evidencia que lesões de Grau 1, com bordas mais difusas e menor densidade de contraste nas imagens FLAIR, impõem um obstáculo geométrico intrínseco aos modelos de segmentação.

---

## 🖼️ Inspeção Qualitativa

O arquivo visual de segmentação comparativa está localizado em:
`latex/exemplo_segmentacao.png`

```
+------------------+---------------------+-------------------+---------------------+-------------------------+
| Imagem FLAIR 2D  | Máscara Referência  | Predição Baseline | Predição Attention  | Predição Att. + Tversky |
| (Entrada MRI)    | (Ground Truth)      | (Dice: 0,85)      | (Dice: 0,83)        | (Maior Recall)          |
+------------------+---------------------+-------------------+---------------------+-------------------------+
```

---

## 🛠️ Estrutura da Pasta e Compilação

```bash
01-segmentacao-tumores-cerebrais-unet/
├── README.md                          # Este documento detalhado
├── artigo_segmentacao_lgg_mri.pdf     # Documento científico final em PDF
└── latex/
    ├── artigo.tex                     # Código-fonte LaTeX IEEEtran
    ├── IEEEtran.cls                   # Classe de documento IEEE
    └── exemplo_segmentacao.png        # Figura qualitativa do artigo
```

### Como Compilar o Artigo em PDF
Requer uma distribuição TeX (ex.: TeX Live, MacTeX) com `pdflatex`:

```bash
cd latex
pdflatex artigo.tex
pdflatex artigo.tex  # Segunda execução para consolidar referências cruzadas
```

---

## 📖 Citação

```bibtex
@inproceedings{silvaneto2026segmentacao,
  author    = {Silva Neto, Jos{\'e} Joaquim da and Gomes, Pedro Henrique},
  title     = {Deep Learning para Segmenta{\c{c}}{\~a}o de Tumores Cerebrais: Aplica{\c{c}}{\~a}o de U-Net no Dataset LGG-MRI},
  booktitle = {Anais da Disciplina de Intelig{\^e}ncia Artificial 2 - UnB FCTE},
  year      = {2026},
  address   = {Bras{\'i}lia, Brasil}
}
```
