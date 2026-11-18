# Coletânea de Pesquisas e Artigos Científicos em Inteligência Artificial

[![Repository](https://img.shields.io/badge/Repositório-GitHub-blue.svg)](https://github.com/phenric26)
[![Instituição](https://img.shields.io/badge/Universidade%20de%20Brasília-UnB%20FCTE-darkblue.svg)](https://fcte.unb.br/)
[![Artigos](https://img.shields.io/badge/Artigos-3%20Papers-success.svg)](#-visão-geral-dos-artigos)
[![Licença](https://img.shields.io/badge/Licen%C3%A7a-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

Este repositório reúne três artigos científicos experimentais desenvolvidos na **Faculdade de Ciências e Tecnologias em Engenharia (FCTE)** da **Universidade de Brasília (UnB)**. 

Os trabalhos abrangem diferentes frentes da Inteligência Artificial moderna — **Visão Computacional Médica**, **Modelos Generativos Adversários (GANs)** e **Aprendizado por Reforço Profundo (DRL)** — unidos por um compromisso comum com o **rigor metodológico, reprodutibilidade, controle de vazamento de dados (*data leakage*), inferência estatística com intervalos de confiança e auditoria de equidade (*fairness*)**.

---

## 👥 Autores e Afiliação

- **José Joaquim da Silva Neto**  
  *Faculdade de Ciências e Tecnologias em Engenharia — Universidade de Brasília (UnB)*  
  Email: netojoaquimjj@gmail.com | Matrícula: 232027510  

- **Pedro Henrique Gomes**  
  *Faculdade de Ciências e Tecnologias em Engenharia — Universidade de Brasília (UnB)*  
  Email: pedrohg2609@gmail.com | Matrícula: 232030041 | GitHub: [@phenric26](https://github.com/phenric26)  

---

## 📚 Visão Geral dos Artigos

```
artigos-ia/
│
├── 📁 01-segmentacao-tumores-cerebrais-unet/
│   ├── 📄 artigo_segmentacao_lgg_mri.pdf
│   ├── 📘 README.md (Documentação completa)
│   └── 📂 latex/ (Código-fonte TeX, IEEEtran.cls e figuras)
│
├── 📁 02-dcgan-dermatofibroma-isic2019/
│   ├── 📄 artigo_dcgan_dermatofibroma.pdf
│   ├── 📘 README.md (Documentação completa)
│   └── 📂 latex/ (Código-fonte TeX, IEEEtran.cls e figuras)
│
└── 📁 03-dqn-ppo-lunarlander-v3/
    ├── 📄 artigo_dqn_ppo_lunarlander.pdf
    ├── 📘 README.md (Documentação completa)
    └── 📂 latex/ (Código-fonte TeX e figuras)
```

---

## 📑 Resumo Comparativo das Pesquisas

| Artigo | Domínio / Tarefa | Dataset / Ambiente | Métodos / Arquiteturas | Descoberta Central |
| :--- | :--- | :--- | :--- | :--- |
| **[Artigo 1: Segmentação de Tumores Cerebrais](01-segmentacao-tumores-cerebrais-unet/)** | Visão Computacional Médica / Segmentação Semântica | TCGA LGG-MRI (110 pacientes, fatias FLAIR 2D, 5 instituições) | • U-Net (ResNet-34)<br>• Attention U-Net<br>• Attention U-Net + Tversky Loss | **Métricas agregadas mascaram o erro clínico:** o Dice geral de 0,852 esconde um Dice de 0,72–0,74 nas fatias com tumor. A perda Tversky atinge o maior recall (0,8116). A análise de equidade revelou disparidade consistente no grau histológico (Grau 1 tem Dice 7 p.p. menor que Grau 2). |
| **[Artigo 2: DCGAN para Dermatofibroma](02-dcgan-dermatofibroma-isic2019/)** | Modelos Generativos / Desbalanceamento Extremo / Equidade | ISIC 2019 (25.331 imagens dermatoscópicas, 239 DF = 0,9%) | • DCGAN (DiffAugment, EMA, Label Smoothing)<br>• ResNet-18 (7 condições)<br>• Bootstrap 1000x | **Resultado negativo robusto:** Imagens sintéticas de DCGAN não melhoraram o recall (+0,008, IC $[-0,026; +0,040]$). Réplicas reais (+0,071) e ponderação na perda (+0,061) superaram a GAN, provando que aumentos atribuídos a dados sintéticos na literatura decorrem do mero rebalanceamento. Disparidade persistente de gênero (recall 0,62 em mulheres vs. 0,47 em homens). |
| **[Artigo 3: DQN vs. PPO no LunarLander-v3](03-dqn-ppo-lunarlander-v3/)** | Aprendizado por Reforço Profundo (DRL) / Controle | LunarLander-v3 (Gymnasium 1.4.0, ações discretas, CPU) | • DQN (*off-policy*)<br>• PPO (*on-policy*)<br>• Estudo de Ablação (R3)<br>• Logger de Entropia e KL | **O Paradoxo do Ajuste e "Fome de Atualizações":** O mesmo conjunto de hiperparâmetros (RL Zoo) que resgatou o DQN da falha total ($-106 \rightarrow +206$, 62,3% sucesso) provocou colapso no PPO ($+108 \rightarrow -165$). O estudo de ablação provou que a causa não foi a taxa de aprendizado, mas a supressão de atualizações de gradiente causada por 16 ambientes paralelos sob 300k passos. |

---

## 🔍 Detalhamento dos Artigos

### 1. [Segmentação de Tumores Cerebrais via U-Net (LGG-MRI)](01-segmentacao-tumores-cerebrais-unet/)
- **Problema:** Delimitação automática de gliomas de baixo grau em ressonâncias magnéticas (FLAIR) sob forte desbalanceamento anatômico e variabilidade institucional.
- **Diferencial:** Particionamento paciente-disjunto (76 treino / 17 validação / 17 teste) sem vazamento entre fatias 2D; calibração cega de limiar; intervalos de confiança bootstrap (95%) por paciente.
- **Link Direto:** [Acessar Pasta do Artigo 1](01-segmentacao-tumores-cerebrais-unet/) | [PDF](01-segmentacao-tumores-cerebrais-unet/artigo_segmentacao_lgg_mri.pdf)

### 2. [Avaliação Controlada de DCGAN no ISIC 2019 (Dermatofibroma)](02-dcgan-dermatofibroma-isic2019/)
- **Problema:** Verificar se dados sintéticos de DCGAN auxiliam um classificador ResNet-18 a detectar a patologia mais rara do ISIC 2019 (0,9% das amostras).
- **Diferencial:** Validação cruzada em 5 folds com a GAN treinada do zero em cada fold usando unicamente as imagens de treino do respectivo fold (zero vazamento); 7 condições experimentais com controles estritos de rebalanceamento; auditoria de memorização via vizinhos mais próximos; análise demográfica de gênero e faixa etária.
- **Link Direto:** [Acessar Pasta do Artigo 2](02-dcgan-dermatofibroma-isic2019/) | [PDF](02-dcgan-dermatofibroma-isic2019/artigo_dcgan_dermatofibroma.pdf)

### 3. [Eficiência Amostral e Estabilidade de DQN e PPO no LunarLander-v3](03-dqn-ppo-lunarlander-v3/)
- **Problema:** Investigar a crise de reprodutibilidade em DRL e diagnosticar a sensibilidade de algoritmos off-policy vs. on-policy sob orçamento restrito de 300.000 passos em CPU.
- **Diferencial:** Comparação entre padrões da biblioteca (R1), parâmetros otimizados da literatura (R2) e ablação isolada de paralelismo (R3); extração dinâmica de métricas internas da rede neural (Entropia da Política e Divergência KL) para comprovar a mecânica do *policy collapse*; testes inferenciais de Welch ($n=5$) e Cohen's $d$.
- **Link Direto:** [Acessar Pasta do Artigo 3](03-dqn-ppo-lunarlander-v3/) | [PDF](03-dqn-ppo-lunarlander-v3/artigo_dqn_ppo_lunarlander.pdf)

---

## 🛠️ Como Compilar os Artigos em LaTeX

Todos os artigos foram diagramados no padrão **IEEEtran** (conferência em duas colunas) e são compiláveis com qualquer distribuição TeX padrão (`texlive-full`, `mactex` ou `miktex`).

Para compilar qualquer um dos artigos:

```bash
# Exemplo para o Artigo 1:
cd 01-segmentacao-tumores-cerebrais-unet/latex
pdflatex artigo.tex
pdflatex artigo.tex

# Exemplo para o Artigo 2:
cd ../../02-dcgan-dermatofibroma-isic2019/latex
pdflatex artigo.tex
pdflatex artigo.tex

# Exemplo para o Artigo 3:
cd ../../03-dqn-ppo-lunarlander-v3/latex
pdflatex artigo.tex
pdflatex artigo.tex
```

---

## 📜 Citação dos Trabalhos

Caso utilize qualquer parte dos dados, análises ou metodologias apresentadas:

```bibtex
@misc{silvaneto_gomes_2026_artigos_ia,
  author       = {Jos{\'e} Joaquim da Silva Neto and Pedro Henrique Gomes},
  title        = {Colet{\^a}nea de Pesquisas e Artigos Cient{\'i}ficos em Intelig{\^e}ncia Artificial: Segmenta{\c{c}}{\~a}o M{\'e}dica, Redes Generativas e Aprendizado por Refor{\c{c}}o},
  institution  = {Universidade de Bras{\'i}lia (UnB), Faculdade de Ci{\^e}ncias e Tecnologias em Engenharia (FCTE)},
  year         = {2026},
  url          = {https://github.com/phenric26}
}
```
