# Imagens Sintéticas de uma DCGAN Ajudam a Detectar Dermatofibroma? Uma Avaliação Controlada no ISIC 2019

[![Paper PDF](https://img.shields.io/badge/Artigo-PDF-red.svg)](artigo_dcgan_dermatofibroma.pdf)
[![LaTeX](https://img.shields.io/badge/LaTeX-IEEEtran-blue.svg)](latex/artigo.tex)
[![Dataset](https://img.shields.io/badge/Dataset-ISIC--2019-orange.svg)](https://challenge.isic-archive.com/landing/2019/)
[![Instituição](https://img.shields.io/badge/UnB-FCTE-darkblue.svg)](https://fcte.unb.br/)

> **Autores:**  
> - **José Joaquim da Silva Neto** (netojoaquimjj@gmail.com | Matrícula: 232027510)  
> - **Pedro Henrique Gomes** (pedrohg2609@gmail.com | Matrícula: 232030041)  
>  
> *Faculdade de Ciências e Tecnologias em Engenharia — Universidade de Brasília (UnB), Brasil*

---

## 📌 Sumário Executivo e Pergunta Científica

O diagnóstico assistido por computador em dermatologia enfrenta um obstáculo crítico de **desbalanceamento extremo de classes**: lesões benignas comuns (como nevos melanocíticos) representam a esmagadora maioria das imagens, enquanto patologias raras aparecem em pouquíssimos exemplares. Um modelo treinado sob tal assimetria tende a ignorar as classes minoritárias, transferindo o risco de erro clínico aos pacientes acometidos por condições atípicas.

Uma solução frequentemente proposta na literatura é o **aumento generativo de dados via GANs** (*Generative Adversarial Networks*). No entanto, muitos estudos não empregam grupos de controle estritos de rebalanceamento, correndo o risco de atribuir às imagens sintéticas ganhos que advêm meramente da reamostragem estatística.

Este artigo investiga experimentalmente:  
> **"Imagens sintéticas geradas por uma DCGAN realmente auxiliam um classificador ResNet-18 a detectar o dermatofibroma (DF) — a classe mais rara do ISIC 2019 — quando comparadas a controles rigorosos de rebalanceamento?"**

---

## 🔬 Protocolo Experimental Rigoroso

### 1. Base de Dados ISIC 2019 e a Minoria Extrema
- **Total de Imagens:** 25.331 dermatoscopias distribuídas em 8 patologias.
- **Classe Majoritária (NV - Nevo Melanocítico):** 12.875 imagens (50,8%).
- **Classe Minoritária Investigada (DF - Dermatofibroma):** apenas **239 imagens (0,9% do total)**.

### 2. Desenho Experimental à Prova de Vazamento (*Data Leakage*)
- **Validação Cruzada Estratificada em 5 Folds:** divisão do rótulo DF contra o resto.
- **Particionamento Triplo:** em cada *fold*, o conjunto de treino foi subdividido para reservar 1/6 como conjunto de validação independente (~16.900 treino, 3.400 validação e 5.070 teste).
- **Treinamento Independente da GAN:** uma instância de DCGAN foi **treinada do zero em cada um dos 5 folds**, utilizando única e exclusivamente as 154 a 163 imagens de DF do conjunto de treino daquele *fold*. As imagens de validação e teste jamais foram vistas pela GAN.
- **Seleção Cega de Hiperparâmetros:** a época do classificador foi selecionada pela máxima PR-AUC na validação, e o limiar de decisão binarizado foi ajustado pelo F1-Score da validação. O conjunto de teste foi avaliado uma única vez após essas definições.

### 3. Arquitetura da DCGAN e Estabilização em Dados Escassos
- **Gerador:** Vetor latente $z \in \mathbb{R}^{100}$ projetado por 5 camadas `ConvTranspose2d` $4\times 4$ (512, 256, 128, 64, 3 canais) com *BatchNorm*, ativações ReLU e saída Tanh em $[-1, 1]$ ($64\times 64\times 3$).
- **Discriminador:** 5 camadas `Conv2d` com *stride* 2 (64, 128, 256, 512, 1 canal de *logit*) com *BatchNorm*, LeakyReLU ($\alpha=0{,}2$) e `BCEWithLogitsLoss`.
- **Técnicas de Estabilização:**
  1. Suavização de rótulos reais (*One-sided Label Smoothing*) para 0,9;
  2. Aumento de dados diferenciável (**DiffAugment**) com transformações de cor, translação e *cutout* aplicadas a imagens reais e falsas antes da retropropagação do discriminador;
  3. Média móvel exponencial (**EMA**) dos pesos do gerador com coeficiente $0{,}999$.
- **Critério de Parada:** Seleção do ponto de menor **FID** (*Fréchet Inception Distance*) medido a cada 250 iterações sobre 1.000 amostras sintéticas.

### 4. As 7 Condições Experimentais do Classificador (ResNet-18)
Para isolar a contribuição das imagens sintéticas, o classificador binário (ResNet-18) foi treinado em sete configurações controladas, executando **3 sementes aleatórias por fold** ($5 \times 3 = 15$ execuções por condição):
1. **Baseline:** Dados originais sem rebalanceamento.
2. **Controle:** Adição de 1.000 réplicas simples de imagens reais de DF.
3. **+ GAN:** Adição de 1.000 imagens sintéticas produzidas pela DCGAN.
4. **Controle + GAN:** Combinação de 1.000 réplicas reais + 1.000 sintéticas.
5. **Peso na Perda:** Ponderação positiva na função de custo ($\sqrt{n_{\text{neg}}/n_{\text{pos}}}$).
6. **Amostragem:** Amostragem balanceada por classe via *WeightedRandomSampler*.
7. **Amostragem + GAN:** Amostragem balanceada suprida com 1.000 sintéticas.

---

## 📊 Resultados Experimentais

### Desempenho Global no Teste (25.331 Imagens, 239 DF)
*Médias consolidadas de 3 sementes aleatórias por fold:*

| Condição Experimental | Recall (Sensib.) | Precisão | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | 0,544 | 0,674 | 0,602 | 0,949 | 0,621 |
| **Controle (Réplicas Reais)** | **0,615** | 0,641 | 0,627 | 0,925 | **0,653** |
| **+ GAN (Sintéticas)** | 0,552 | 0,627 | 0,587 | 0,948 | 0,611 |
| **Controle + GAN** | 0,589 | **0,686** | **0,632** | 0,913 | 0,636 |
| **Peso na Perda** | 0,605 | 0,622 | 0,612 | **0,959** | 0,618 |
| **Amostragem Balanceada** | 0,565 | 0,620 | 0,591 | 0,917 | 0,605 |
| **Amostragem + GAN** | 0,552 | 0,658 | 0,599 | 0,927 | 0,605 |

### Diferenças Pareadas com Intervalo de Confiança Bootstrap de 95%
*1.000 reamostragens estratificadas por imagem. Valores em negrito indicam intervalos que excluem o zero (estatisticamente significantes):*

| Comparação Pareada ($A - B$) | Diferença de Recall (IC 95%) | Diferença de PR-AUC (IC 95%) |
| :--- | :---: | :---: |
| **+GAN $-$ Baseline** | $+0,008$ ($-0,026$ a $+0,040$) | $-0,009$ ($-0,036$ a $+0,017$) |
| **Controle+GAN $-$ Controle** | $-0,027$ ($-0,056$ a $+0,001$) | $-0,016$ ($-0,039$ a $+0,005$) |
| **Controle $-$ Baseline** | **$+0,071$ ($+0,036$ a $+0,109$)** | **$+0,032$ ($+0,005$ a $+0,058$)** |
| **Peso na Perda $-$ Baseline** | **$+0,061$ ($+0,031$ a $+0,091$)** | $-0,003$ ($-0,024$ a $+0,017$) |

---

## 💡 Principais Descobertas e Discussão

### 1. O Resultado Negativo Robusto das Imagens Sintéticas
- A adição de 1.000 imagens da DCGAN isolada (**+GAN**) produziu uma variação irrisória de recall de apenas **$+0,008$** com IC 95% variando de **$-0,026$ a $+0,040$** (abrangendo o zero), acompanhada de uma **queda de 4,9 pontos percentuais na precisão**.
- Quando combinadas com réplicas ou amostragem balanceada, as imagens sintéticas não trouxeram benefício adicional demonstrável.

### 2. O Valor Crucial dos Controles de Rebalanceamento
- A simples repetição de imagens dermatoscópicas reais já existentes (**Controle**) aumentou o recall em **$+0,071$** (IC $[+0,036; +0,109]$) e a PR-AUC em $+0,032$.
- A atribuição de **pesos maiores na função de perda** elevou o recall em **$+0,061$** com a maior ROC-AUC de todas as condições ($0,959$).
- *Conclusão Metodológica:* Sem os devidos grupos de controle, pesquisadores poderiam erroneamente atribuir às imagens da GAN ganhos de sensibilidade que, na realidade, decorrem unicamente do rebalanceamento da taxa de exposição aos dados.

### 3. Análise da GAN: Fidelidade vs. Diversidade
- O **FID** médio obtido variou entre **191 e 225** contra as amostras reais, refletindo imagens com tons de pele e máculas centrais realistas, mas desprovidas de microestruturas dermatológicas finas (ex.: rede pigmentar e estruturas estriadas periféricas).
- O teste de vizinhos mais próximos no espaço latente ResNet-18 descartou memorização direta (mediana de distância 8,38 para sintéticas vs. 7,36 para imagens reais não vistas). Entretanto, revelou **baixa diversidade**: múltiplas sintéticas mapeavam para uma mesma imagem real de apoio.

---

## ⚖️ Equidade e Fairness Demográfico

Utilizando os metadados oficiais do ISIC (sexo, faixa etária e topografia corporal), mediu-se o recall por subgrupo:

| Subgrupo | Amostra DF | Recall Baseline | Recall Controle | Recall + GAN | Recall Peso na Perda |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Sexo: Feminino** | 117 | **0,62** | 0,69 | 0,63 | 0,69 |
| **Sexo: Masculino** | 122 | **0,47** | 0,54 | 0,48 | 0,52 |
| **Idade: < 40 anos** | 39 | 0,65 | 0,68 | 0,67 | 0,68 |
| **Idade: 40 a 59 anos** | 121 | **0,47** | 0,56 | 0,49 | 0,55 |
| **Idade: $\geq$ 60 anos** | 79 | 0,61 | 0,66 | 0,59 | 0,65 |
| **Membros Inferiores (MI)** | 138 | 0,57 | 0,63 | 0,59 | 0,61 |
| **Membros Superiores (MS)** | 52 | 0,49 | 0,60 | 0,49 | 0,54 |
| **Tronco** | 45 | 0,57 | 0,60 | 0,53 | 0,61 |

> **Disparidade de Gênero Persistente:**  
> O modelo detecta significativamente melhor o dermatofibroma em **mulheres (recall 0,62)** do que em **homens (recall 0,47)** no baseline — uma disparidade de 14,6 pontos percentuais (IC 95%: $[0,036; 0,249]$). Nenhuma estratégia generativa ou de controle foi capaz de extinguir essa assimetria.

---

## 🖼️ Figuras e Ativos Experimentais

- `latex/fig_grids.png`: Matriz comparativa entre imagens reais de DF (à esquerda) e imagens sintéticas geradas pela DCGAN (à direita).
- `latex/fig_loss.pdf`: Curvas de convergência da perda do discriminador e gerador ao longo de 4.000 iterações adversárias.
- `latex/fig_nn.png`: Mapeamento do vizinho mais próximo (*Nearest Neighbor*) demonstrando fidelidade visual e concentração no espaço latente.

---

## 🛠️ Estrutura da Pasta e Compilação

```bash
02-dcgan-dermatofibroma-isic2019/
├── README.md                          # Documentação científica detalhada
├── artigo_dcgan_dermatofibroma.pdf    # PDF final do artigo compilado
└── latex/
    ├── artigo.tex                     # Código-fonte LaTeX IEEEtran
    ├── IEEEtran.cls                   # Classe IEEE
    ├── fig_grids.png                  # Amostras reais vs sintéticas
    ├── fig_loss.pdf                   # Gráfico de perdas da DCGAN
    └── fig_nn.png                     # Análise de vizinho mais próximo
```

### Como Compilar o Artigo em PDF
```bash
cd latex
pdflatex artigo.tex
pdflatex artigo.tex
```

---

## 📖 Citação

```bibtex
@inproceedings{silvaneto2026dcgan,
  author    = {Silva Neto, Jos{\'e} Joaquim da and Gomes, Pedro Henrique},
  title     = {Imagens Sint{\'e}ticas de uma DCGAN Ajudam a Detectar Dermatofibroma? Uma Avalia{\c{c}}{\~a}o Controlada no ISIC 2019},
  booktitle = {Anais da Disciplina de Intelig{\^e}ncia Artificial 2 - UnB FCTE},
  year      = {2026},
  address   = {Bras{\'i}lia, Brasil}
}
```
