# Eficiência Amostral e Estabilidade de DQN e PPO em Navegação Autônoma: Um Estudo de Ablação no LunarLander-v3

[![Paper PDF](https://img.shields.io/badge/Artigo-PDF-red.svg)](artigo_dqn_ppo_lunarlander.pdf)
[![LaTeX](https://img.shields.io/badge/LaTeX-IEEEtran-blue.svg)](latex/artigo.tex)
[![Gymnasium](https://img.shields.io/badge/Environment-LunarLander--v3-brightgreen.svg)](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
[![Stable-Baselines3](https://img.shields.io/badge/Framework-Stable--Baselines3-blueviolet.svg)](https://stable-baselines3.readthedocs.io/)
[![Instituição](https://img.shields.io/badge/UnB-FCTE-darkblue.svg)](https://fcte.unb.br/)

**Autores:**  
- José Joaquim da Silva Neto (netojoaquimjj@gmail.com | Matrícula: 232027510)  
- Pedro Henrique Gomes (pedrohg2609@gmail.com | Matrícula: 232030041)  

*Faculdade de Ciências e Tecnologias em Engenharia — Universidade de Brasília (UnB), Brasil*

---

## Sumário Executivo

A literatura de Aprendizado por Reforço Profundo (DRL - *Deep Reinforcement Learning*) enfrenta desafios consolidados de reprodutibilidade empírica: os modelos exibem alta sensibilidade a sementes pseudoaleatórias, escalonamento de paralelismo e configurações hiperparamétricas. A popularização de bibliotecas abstratas (como o *Stable-Baselines3*) propagou a suposição de que hiperparâmetros padrão ou extraídos de repositórios consolidados (*RL Baselines3 Zoo*) comportam-se de forma universal sob qualquer orçamento computacional.

Este estudo conduz uma análise comparativa sistemática entre dois algoritmos representativos de paradigmas teóricos distintos:
- **DQN (*Deep Q-Network*):** Método baseado em valor, fora da política (*off-policy*), que armazena e reutiliza experiências por meio de um *buffer* de repetição;
- **PPO (*Proximal Policy Optimization*):** Método de gradiente de política, dentro da política (*on-policy*), com recorte objetivo (*clipping*) na atualização de parâmetros.

A avaliação foi realizada sob um orçamento rígido de **300.000 passos de ambiente em CPU** no ambiente de controle contínuo discreto **LunarLander-v3**. A metodologia integra curvas de convergência, avaliação cega final ($n=5$ sementes), testes $t$ de Welch com $d$ de Cohen e um **estudo de ablação munido de instrumentação de métricas internas (Entropia da Política e Divergência KL)** para caracterizar o fenômeno de colapso de políticas (*policy collapse*).

---

## Questões Científicas Centrais

1. **Eficiência Amostral vs. Tempo de Processamento (*Wall-Clock Time*):** Sob um teto de 300 mil passos, qual algoritmo atinge convergência satisfatória e qual apresenta menor tempo de relógio?
2. **Impacto do Ajuste Hiperparamétrico Global:** Como cada abordagem reage quando submetida a um conjunto de hiperparâmetros ajustados da literatura (*RL Zoo*) em relação aos valores padrão?
3. **Mecânica do Colapso de Política:** Por meio de ablação de controle e monitoramento interno, qual é a contribuição estrita da paralelização de ambientes ($n\_envs$) na instabilidade de algoritmos *on-policy*?

---

## Metodologia e Desenho Experimental

### 1. Ambiente e Critério de Resolução
- **Ambiente:** `LunarLander-v3` da biblioteca Gymnasium (1.4.0) com espaço de ações discretas.
- **Critério de Sucesso:** Retorno acumulado por episódio $\geq +200$ pontos, com truncamento em 1.000 passos.
- **Orçamento Computacional:** Limite superior rígido de 300.000 passos de ambiente executados integralmente em CPU.

### 2. Regimes Experimentais Avaliados

```
+-------------------------------------------------------------------------------------------------+
|                                 CONFIGURAÇÕES EXPERIMENTAIS                                     |
+-----------------------------------+-----------------------------------+-------------------------+
| R1 (Baseline Padrão)              | R2 (Sintonia Global - RL Zoo)     | R3 (Ablação de Controle)|
| - Parâmetros padrão do SB3        | - Hiperparâmetros recomendados    | - Parâmetros padrão R1  |
| - 1 ambiente único (n_envs = 1)   | - DQN: buffer 50k, MLP [256, 256] | - APENAS n_envs = 16    |
| - Avalia a resiliência original   | - PPO: n_envs = 16, gamma = 0,999 | - Isola o paralelismo   |
|   de cada algoritmo               | - Avalia o pipeline da literatura |   na mecânica do PPO    |
+-----------------------------------+-----------------------------------+-------------------------+
```

### 3. Tabela de Hiperparâmetros (R2 - RL Zoo)

| Hiperparâmetro | DQN (R2) | PPO (R2) |
| :--- | :---: | :---: |
| **Ambientes Paralelos ($n\_envs$)** | 1 | **16** |
| **Taxa de Aprendizado (Adam)** | $6{,}3 \times 10^{-4}$ | $3 \times 10^{-4}$ (padrão) |
| **Tamanho do Batch** | 128 | 64 |
| **Fator de Desconto ($\gamma$)** | 0,99 | 0,999 |
| **Buffer de Repetição / Início do Treino** | 50.000 / 0 passos | — |
| **Frequência de Atualização da Rede Alvo** | 250 passos | — |
| **Frequência de Treino / Passos de Gradiente** | 4 / $-1$ | — |
| **Fração de Exploração / $\epsilon$ Final** | 0,12 / 0,1 | — |
| **Camadas Ocultas (MLP)** | $[256, 256]$ (ReLU) | $[64, 64]$ (ReLU padrão) |
| **Passos por Rollout ($n\_steps$) / Épocas** | — | 1024 / 4 |
| **Parâmetro GAE ($\lambda$) / Coef. Entropia** | — | 0,98 / 0,01 |

### 4. Protocolo Estatístico e Instrumentação
- **Sementes Independentes ($n=5$):** 10, 42, 100, 7 e 2024.
- **Avaliações Periódicas:** A cada 10.000 passos, a política foi congelada e submetida a 10 episódios determinísticos.
- **Teste Cego Final:** O modelo final treinado foi avaliado ao longo de **100 episódios de teste por semente** ($5 \times 100 = 500$ episódios no total por configuração).
- **Métricas Internas:** Coleta contínua de **Entropia da Política** (`policy_entropy`) e **Divergência KL Aproximada** (`approx_kl`) a cada passo de otimização de gradiente.

---

## Resultados Experimentais

### Desempenho no Teste Cego Final (Média $\pm$ Desvio-Padrão para $n=5$)

| Configuração Experimental | Retorno Médio (100 ep.) | Taxa de Sucesso (%) | Tempo de Execução (min) |
| :--- | :---: | :---: | :---: |
| **DQN R1** (Baseline, 1 env) | $-106{,}7 \pm 3{,}2$ | $0{,}0\%$ | 21,1 min |
| **PPO R1** (Baseline, 1 env) | $+108{,}8 \pm 58{,}4$ | $25{,}3\%$ | 15,4 min |
| **DQN R2** (Sintonizado, 1 env) | **$+206{,}4 \pm 34{,}3$** | **$62{,}3\%$** | 28,4 min |
| **PPO R2** (Sintonizado, 16 envs) | $-165{,}3 \pm 1{,}1$ | $0{,}0\%$ | **7,5 min** |
| **PPO R3** (Ablação de Controle, 16 envs) | $+50{,}3 \pm 71{,}2$ | $12{,}5\%$ | 14,1 min |

### Validação Inferencial: Testes $t$ de Welch ($n=5$)

| Comparação Analisada | Estatística $t$ | $p$-valor | $d$ de Cohen | Intervalo de Confiança (95%) |
| :--- | :---: | :---: | :---: | :---: |
| **R1: DQN vs. PPO** | $-6{,}38$ | $0{,}0234$ | $-5{,}21$ | $[-360{,}8;\; -70{,}2]$ |
| **R2: DQN vs. PPO** | $+18{,}76$ | $0{,}0028$ | $+15{,}32$ | $[+286{,}5;\; +456{,}9]$ |
| **DQN: Baseline (R1) vs. Sintonizado (R2)** | $-15{,}74$ | $0{,}0037$ | $-12{,}85$ | $[-398{,}7;\; -227{,}5]$ |
| **PPO: Baseline (R1) vs. Sintonizado (R2)** | $+8{,}12$ | $0{,}0148$ | $+6{,}63$ | $[+129{,}1;\; +419{,}1]$ |

---

## Discussão e Achados Científicos

### 1. O Paradoxo do Ajuste
Os resultados comprovam a dependência estrita dos hiperparâmetros em relação às condições de amostragem:
- O **DQN** sob valores padrão (R1) falhou em convergir (retorno negativo persistente de $-106{,}7$). A sintonia estrutural (R2) promoveu a resolução da tarefa, atingindo **$+206{,}4$ pontos** e **62,3% de taxa de sucesso**.
- O **PPO** exibiu comportamento inverso: convergiu adequadamente no padrão R1 ($+108{,}8$), mas sofreu **degradação severa na configuração R2 ($-165{,}3$)**.

### 2. Diagnóstico de Colapso via Ablação (R3)
O grupo de controle de ablação (**R3**) permitiu isolar a causa da falha do PPO. Ao reter os hiperparâmetros padrão da R1 e alterar apenas o grau de paralelização para $n\_envs = 16$, observou-se desempenho instável ($+50{,}3 \pm 71{,}2$).

O mecanismo responsável é a **escassez de atualizações de gradiente**:
1. No PPO, os parâmetros são atualizados após o acúmulo de $n\_steps \times n\_envs$ transições. Com $1024 \times 16$, cada ciclo exige **16.384 passos**.
2. Sob um teto de 300.000 passos, o algoritmo realiza apenas cerca de **18 ciclos de gradiente** em todo o treinamento.
3. O monitoramento das variáveis internas (`latex/figs/raio_x.png`) comprovou que a **Entropia da Política decaiu prematuramente**, enquanto a **Divergência KL oscilou com picos abruptos** ($> 0{,}016$). Sem atualizações em frequência suficiente, a política convergiu precocemente para atratores subótimos de navegação (*Policy Collapse*).

### 3. O Trade-Off: Eficiência de Amostras vs. Wall-Clock Time
- **DQN (R2):** Apresentou alta eficiência amostral, solucionando a tarefa dentro dos 300.000 passos virtuais. Como opera em processo único (*single-thread*), demandou **28,4 minutos**.
- **PPO (R2):** Apresentou alta vazão computacional, completando os 300 mil passos em **7,5 minutos** devido aos 16 processos paralelos. Todavia, sob volume restrito de passos, a paralelização excessiva compromete a estabilidade matemática do algoritmo.

---

## Figuras e Gráficos do Estudo

- `latex/figs/curvas_avaliacao.png`: Curvas dinâmicas de evolução na Rodada 1 (*Baseline*).
- `latex/figs/raio_x.png`: Diagnóstico de colapso do PPO R2 via Entropia da Política e Divergência KL.
- `latex/figs/fig_curvas.png` e `latex/figs/fig_final.png`: Gráficos consolidados do teste final.

---

## Estrutura de Diretórios e Compilação

```
03-dqn-ppo-lunarlander-v3/
├── README.md
├── artigo_dqn_ppo_lunarlander.pdf
└── latex/
    ├── artigo.tex
    └── figs/
        ├── curvas_avaliacao.png
        ├── raio_x.png
        ├── fig_curvas.png
        ├── fig_curvas.pdf
        ├── fig_final.png
        └── fig_final.pdf
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
@inproceedings{silvaneto2026lunarlander,
  author    = {Silva Neto, Jos{\'e} Joaquim da and Gomes, Pedro Henrique},
  title     = {Efici{\^e}ncia Amostral e Estabilidade de DQN e PPO em Navega{\c{c}}{\~a}o Aut{\^o}noma: Um Estudo de Abla{\c{c}}{\~a}o no LunarLander-v3},
  booktitle = {Anais da Disciplina de Intelig{\^e}ncia Artificial 2 - UnB FCTE},
  year      = {2026},
  address   = {Bras{\'i}lia, Brasil}
}
```
