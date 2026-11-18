# Imagens Sintéticas de uma DCGAN Ajudam a Detectar Dermatofibroma? Uma Avaliação Controlada no ISIC 2019

[![Paper PDF](https://img.shields.io/badge/Artigo-PDF-red.svg)](artigo_dcgan_dermatofibroma.pdf)
[![LaTeX](https://img.shields.io/badge/LaTeX-IEEEtran-blue.svg)](latex/artigo.tex)
[![Dataset](https://img.shields.io/badge/Dataset-ISIC--2019-orange.svg)](https://challenge.isic-archive.com/landing/2019/)
[![Instituição](https://img.shields.io/badge/UnB-FCTE-darkblue.svg)](https://fcte.unb.br/)

**Autores:**  
- José Joaquim da Silva Neto (netojoaquimjj@gmail.com | Matrícula: 232027510)  
- Pedro Henrique Gomes (pedrohg2609@gmail.com | Matrícula: 232030041)  

*Faculdade de Ciências e Tecnologias em Engenharia — Universidade de Brasília (UnB), Brasil*

---

## Sumário Executivo e Pergunta Científica

O diagnóstico dermatológico assistido por inteligência artificial depende de bases de dados fortemente desbalanceadas: lesões prevalentes concentram dezenas de milhares de amostras, enquanto patologias raras estão restritas a poucos exemplos. Modelos treinados sob esse regime tendem a apresentar taxa de detecção reduzida na minoria clínica, comprometendo a equidade do atendimento diagnóstico.

O aumento de dados generativo por meio de Redes Adversárias Generativas (GANs) é frequentemente proposto como alternativa de mitigação. Entretanto, publicações na literatura muitas vezes omitem grupos de controle rigorosos para isolar os efeitos do rebalanceamento estatístico simples em comparação à contribuição efetiva das imagens sintéticas.

Este estudo avalia empiricamente:  
> **"Imagens sintéticas geradas por uma DCGAN auxiliam um classificador ResNet-18 a detectar o dermatofibroma (DF) — a classe mais rara do ISIC 2019 — quando submetidas a um protocolo controlado com grupos de rebalanceamento?"**

---

## Protocolo Experimental

### 1. Base de Dados ISIC 2019
- **Amostra Total:** 25.331 dermatoscopias distribuídas em 8 classes diagnósticas.
- **Classe Majoritária (NV - Nevo Melanocítico):** 12.875 imagens (50,8%).
- **Classe Minoritária Investigada (DF - Dermatofibroma):** 239 imagens (0,9% do total).

### 2. Desenho Experimental e Mitigação de Vazamento
- **Validação Cruzada Estratificada em 5 Folds:** Avaliação binária de DF contra as demais lesões.
- **Divisão Interna:** Em cada *fold*, o conjunto de treino foi subdividido para separar 1/6 como conjunto de validação independente (~16.900 treino, 3.400 validação e 5.070 teste). Cada imagem do conjunto total figurou no teste exatamente uma vez.
- **Treinamento Independente da DCGAN:** Uma rede generativa independente foi **treinada do zero em cada fold**, utilizando exclusivamente as 154 a 163 imagens de treino de DF da partição correspondente. Fatias de validação e teste jamais foram disponibilizadas ao gerador ou discriminador.
- **Critério de Decisão:** A época final de treino do classificador foi selecionada pela máxima PR-AUC no conjunto de validação, e o limiar de corte binarizado foi definido pelo F1-Score na validação. O teste final foi conduzido de modo estritamente cego.

### 3. Arquitetura da DCGAN e Estabilização em Dados Reduzidos
- **Gerador:** Vetor latente $z \in \mathbb{R}^{100}$ com cinco camadas `ConvTranspose2d` $4\times 4$ (512, 256, 128, 64, 3 canais), normalização em lote (*BatchNorm*), ReLU e camada final Tanh em $[-1, 1]$ gerando imagens $64\times 64\times 3$.
- **Discriminador:** Cinco camadas `Conv2d` com *stride* 2 (64, 128, 256, 512, 1 canal de *logit*), *BatchNorm*, LeakyReLU ($\alpha=0{,}2$) e `BCEWithLogitsLoss`.
- **Mecanismos de Estabilização:**
  1. Suavização de rótulos reais (*One-sided Label Smoothing*) para 0,9;
  2. Aumento de dados diferenciável (**DiffAugment**) com perturbações de cor, translação e *cutout*;
  3. Média móvel exponencial (**EMA**) dos pesos do gerador com taxa $0{,}999$.
- **Critério de Seleção:** Ponto de menor **FID** (*Fréchet Inception Distance*) medido a cada 250 iterações sobre 1.000 amostras sintéticas contra as imagens reais de treino.

### 4. Condições Experimentais do Classificador (ResNet-18)
O classificador ResNet-18 operou sob 7 condições controladas, com **3 sementes aleatórias por fold** ($5 \times 3 = 15$ execuções por condição):
1. **Baseline:** Dados originais sem técnicas de rebalanceamento.
2. **Controle:** Adição de 1.000 réplicas de imagens reais de DF já existentes.
3. **+ GAN:** Adição de 1.000 amostras sintéticas produzidas pela DCGAN.
4. **Controle + GAN:** Conjunção de 1.000 réplicas reais e 1.000 sintéticas.
5. **Peso na Perda:** Ponderação na função de custo ($\sqrt{n_{\text{neg}}/n_{\text{pos}}}$).
6. **Amostragem:** Amostragem balanceada por classe (*WeightedRandomSampler*).
7. **Amostragem + GAN:** Amostragem balanceada associada a 1.000 imagens sintéticas.

---

## Resultados Experimentais

### Desempenho Global no Teste (25.331 Imagens, 239 DF)
*Médias consolidadas de 3 sementes aleatórias por fold:*

| Condição Experimental | Recall (Sensibilidade) | Precisão | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | 0,544 | 0,674 | 0,602 | 0,949 | 0,621 |
| **Controle (Réplicas Reais)** | **0,615** | 0,641 | 0,627 | 0,925 | **0,653** |
| **+ GAN (Sintéticas)** | 0,552 | 0,627 | 0,587 | 0,948 | 0,611 |
| **Controle + GAN** | 0,589 | **0,686** | **0,632** | 0,913 | 0,636 |
| **Peso na Perda** | 0,605 | 0,622 | 0,612 | **0,959** | 0,618 |
| **Amostragem Balanceada** | 0,565 | 0,620 | 0,591 | 0,917 | 0,605 |
| **Amostragem + GAN** | 0,552 | 0,658 | 0,599 | 0,927 | 0,605 |

### Diferenças Pareadas com Intervalo de Confiança Bootstrap de 95%
*1.000 reamostragens pareadas. Valores em negrito indicam significância estatística (excluem o zero):*

| Comparação Pareada ($A - B$) | Diferença de Recall (IC 95%) | Diferença de PR-AUC (IC 95%) |
| :--- | :---: | :---: |
| **+GAN $-$ Baseline** | $+0,008$ ($-0,026$ a $+0,040$) | $-0,009$ ($-0,036$ a $+0,017$) |
| **Controle+GAN $-$ Controle** | $-0,027$ ($-0,056$ a $+0,001$) | $-0,016$ ($-0,039$ a $+0,005$) |
| **Controle $-$ Baseline** | **$+0,071$ ($+0,036$ a $+0,109$)** | **$+0,032$ ($+0,005$ a $+0,058$)** |
| **Peso na Perda $-$ Baseline** | **$+0,061$ ($+0,031$ a $+0,091$)** | $-0,003$ ($-0,024$ a $+0,017$) |

---

## Principais Descobertas e Discussão

### 1. Ausência de Ganho Mensurável com Dados Sintéticos
- A introdução isolada de imagens sintéticas (**+GAN**) gerou variação marginal de recall ($+0,008$, IC $[-0,026; +0,040]$), enquanto causou redução de **4,9 pontos percentuais na precisão**.
- Em nenhuma das composições (com réplicas ou com amostragem balanceada) os dados sintéticos produziram acréscimo estatisticamente verificável em relação aos seus controles respectivos.

### 2. A Relevância dos Grupos de Controle
- A simples repetição de imagens reais (**Controle**) ampliou o recall em **$+0,071$** (IC $[+0,036; +0,109]$) e a PR-AUC em $+0,032$.
- A atribuição de **pesos na função de perda** aumentou o recall em **$+0,061$** e manteve a maior ROC-AUC da avaliação ($0,959$).
- Esses resultados indicam que melhorias reportadas em trabalhos semelhantes sem grupos de controle decorrem primordialmente da reamostragem estatística, e não do conteúdo semântico gerado pelo modelo adversarial.

### 3. Análise da DCGAN: Fidelidade e Diversidade
- O **FID** médio situou-se entre **191 e 225**, compatível com imagens que replicam paleta cromática e formato macroscópico, porém carecem de texturas dermatoscópicas finas (ex.: rede pigmentar periférica e despigmentação central).
- O teste de vizinho mais próximo (*Nearest Neighbor*) indicou ausência de memorização direta (mediana de distância 8,38 para sintéticas vs. 7,36 para imagens reais não vistas), mas apontou **baixa diversidade espacial**, com concentração das amostras sintéticas em poucos polos do espaço latente.

---

## Equidade e Fairness Demográfico

Desempenho de recall estratificado por sexo, faixa etária e topografia anatômica:

| Subgrupo | Casos DF | Recall Baseline | Recall Controle | Recall + GAN | Recall Peso na Perda |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Sexo: Feminino** | 117 | **0,62** | 0,69 | 0,63 | 0,69 |
| **Sexo: Masculino** | 122 | **0,47** | 0,54 | 0,48 | 0,52 |
| **Idade: < 40 anos** | 39 | 0,65 | 0,68 | 0,67 | 0,68 |
| **Idade: 40 a 59 anos** | 121 | **0,47** | 0,56 | 0,49 | 0,55 |
| **Idade: $\geq$ 60 anos** | 79 | 0,61 | 0,66 | 0,59 | 0,65 |
| **Membros Inferiores (MI)** | 138 | 0,57 | 0,63 | 0,59 | 0,61 |
| **Membros Superiores (MS)** | 52 | 0,49 | 0,60 | 0,49 | 0,54 |
| **Tronco** | 45 | 0,57 | 0,60 | 0,53 | 0,61 |

A disparidade demográfica mais relevante ocorreu entre os sexos: o classificador demonstrou recall substancialmente superior em **mulheres (0,62)** em relação a **homens (0,47)** no baseline — diferença de 14,6 pontos percentuais (IC 95%: $[0,036; 0,249]$). Nenhuma das abordagens generativas ou de controle conseguiu eliminar essa lacuna.

---

## Ativos Experimentais e Figuras

- `latex/fig_grids.png`: Painel comparativo de amostras reais versus sintéticas.
- `latex/fig_loss.pdf`: Curvas de evolução da perda do gerador e do discriminador.
- `latex/fig_nn.png`: Projeção de vizinhos mais próximos no espaço de atributos.

---

## Estrutura de Diretórios e Compilação

```
02-dcgan-dermatofibroma-isic2019/
├── README.md
├── artigo_dcgan_dermatofibroma.pdf
└── latex/
    ├── artigo.tex
    ├── IEEEtran.cls
    ├── fig_grids.png
    ├── fig_loss.pdf
    └── fig_nn.png
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
@inproceedings{silvaneto2026dcgan,
  author    = {Silva Neto, Jos{\'e} Joaquim da and Gomes, Pedro Henrique},
  title     = {Imagens Sint{\'e}ticas de uma DCGAN Ajudam a Detectar Dermatofibroma? Uma Avalia{\c{c}}{\~a}o Controlada no ISIC 2019},
  booktitle = {Anais da Disciplina de Intelig{\^e}ncia Artificial 2 - UnB FCTE},
  year      = {2026},
  address   = {Bras{\'i}lia, Brasil}
}
```
