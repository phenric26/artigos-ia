# Eficiência Amostral e Estabilidade de DQN e PPO em Navegação Autônoma: Um Estudo de Ablação no LunarLander-v3

[![Paper PDF](https://img.shields.io/badge/Artigo-PDF-red.svg)](artigo_dqn_ppo_lunarlander.pdf)
[![LaTeX](https://img.shields.io/badge/LaTeX-IEEEtran-blue.svg)](latex/artigo.tex)
[![Gymnasium](https://img.shields.io/badge/Environment-LunarLander--v3-brightgreen.svg)](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
[![Stable-Baselines3](https://img.shields.io/badge/Framework-Stable--Baselines3-blueviolet.svg)](https://stable-baselines3.readthedocs.io/)
[![Instituição](https://img.shields.io/badge/UnB-FCTE-darkblue.svg)](https://fcte.unb.br/)

> **Autores:**  
> - **José Joaquim da Silva Neto** (netojoaquimjj@gmail.com | Matrícula: 232027510)  
> - **Pedro Henrique Gomes** (pedrohg2609@gmail.com | Matrícula: 232030041)  
>  
> *Faculdade de Ciências e Tecnologias em Engenharia — Universidade de Brasília (UnB), Brasil*

---

## 📌 Sumário Executivo

A comunidade de Aprendizado por Reforço Profundo (**DRL** - *Deep Reinforcement Learning*) enfrenta uma crise metodológica e de reprodutibilidade amplamente debatida: algoritmos do estado da arte apresentam sensibilidade extrema a sementes aleatórias, esquemas de paralelismo de ambientes e seleção de hiperparâmetros. O uso generalizado de bibliotecas de alto nível (como o *Stable-Baselines3*) propagou o mito de que hiperparâmetros pré-configurados ou retirados de repositórios da literatura (*RL Zoo*) são "caixas-pretas" universalmente transferíveis.

Este estudo realiza uma comparação empírica minuciosa entre dois algoritmos seminais que representam dicotomias teóricas fundamentais:
- **DQN (*Deep Q-Network*):** Método baseado em valor, fora da política (*off-policy*), que reutiliza experiências via *Replay Buffer*;
- **PPO (*Proximal Policy Optimization*):** Método de gradiente de política, dentro da política (*on-policy*), com recorte (*clipping*) da função objetivo.

O experimento foi conduzido sob um teto rígido de **300.000 passos de ambiente em CPU** no ambiente clássico de controle **LunarLander-v3** (ações discretas), combinando curvas dinâmicas de treinamento, teste cego final com 5 sementes ($n=5$), testes estatísticos de Welch com $d$ de Cohen e um **estudo de ablação com instrumentação de métricas internas (Entropia e Divergência KL)** para radiografar o colapso de políticas neurais.

---

## 🎯 As Três Questões Científicas Centrais

1. **Eficiência Amostral vs. Tempo de Relógio (*Wall-Clock Time*):** Sob um teto finito de 300 mil passos, qual arquitetura atinge maior retorno e qual executa mais rápido na prática?
2. **O Paradoxo do Ajuste Global:** Qual é a resposta de cada algoritmo ao receber um conjunto de hiperparâmetros "otimizados" da literatura (*RL Baselines3 Zoo*) em substituição ao *baseline* padrão?
3. **Mecânica do Colapso (*Policy Collapse*):** Através de um grupo de ablação rigoroso e monitoramento em tempo real de entropia e divergência KL, qual é o papel exato do paralelismo de ambientes ($n\_envs$) no colapso de algoritmos *on-policy*?

---

## 🔬 Metodologia e Desenho Experimental

### 1. Ambiente e Critério de Resolução
- **Ambiente:** `LunarLander-v3` da biblioteca Gymnasium (1.4.0) com espaço de ações discretas (motor principal, motores laterais, inércia).
- **Critério de Sucesso:** Retorno acumulado do episódio $\geq +200$ pontos (pouso suave entre as bandeiras com velocidade controlada e sem colisão).
- **Orçamento Computacional:** Limite estrito de 300.000 passos em CPU.

### 2. Os Três Regimes Experimentais Avaliados

```
+-------------------------------------------------------------------------------------------------+
|                                 CONFIGURAÇÕES EXPERIMENTAIS                                     |
+-----------------------------------+-----------------------------------+-------------------------+
| R1 (Baseline Padrão)              | R2 (Sintonia Global - RL Zoo)     | R3 (Ablação de Controle)|
| • Padrões "out-of-the-box" SB3    | • Hiperparâmetros recomendados    | • Parâmetros exatos de  |
| • 1 único ambiente (n_envs = 1)   | • DQN: buffer 50k, MLP [256, 256] |   R1 (default SB3)      |
| • Testa a resiliência original    | • PPO: n_envs = 16, gamma = 0,999 | • APENAS n_envs = 16    |
|   de cada algoritmo               | • Testa o pipeline holístico      | • Isola o paralelismo   |
+-----------------------------------+-----------------------------------+-------------------------+
```

### 3. Tabela Detalhada de Hiperparâmetros (R2 - RL Zoo)

| Hiperparâmetro | DQN (R2) | PPO (R2) |
| :--- | :---: | :---: |
| **Ambientes Paralelos ($n\_envs$)** | 1 | **16** |
| **Taxa de Aprendizado (Adam)** | $6{,}3 \times 10^{-4}$ (constante) | $3 \times 10^{-4}$ (padrão) |
| **Tamanho do Batch** | 128 | 64 |
| **Fator de Desconto ($\gamma$)** | 0,99 | 0,999 |
| **Buffer de Repetição / Início do Treino** | 50.000 / 0 passos | — (On-policy) |
| **Frequência de Atualização da Rede Alvo** | 250 passos | — |
| **Frequência de Treino / Passos de Gradiente** | 4 / $-1$ | — |
| **Fração de Exploração / $\epsilon$ Final** | 0,12 / 0,1 | — |
| **Arquitetura da Rede Neural (MLP)** | $[256, 256]$ (ReLU) | $[64, 64]$ (ReLU padrão) |
| **Passos por Rollout ($n\_steps$) / Épocas** | — | 1024 / 4 |
| **Parâmetro GAE ($\lambda$) / Coef. Entropia** | — | 0,98 / 0,01 |

### 4. Protocolo Estatístico e Instrumentação
- **Sementes Independentes ($n=5$):** 10, 42, 100, 7 e 2024.
- **Avaliações Dinâmicas:** A cada 10.000 passos, a política foi congelada e avaliada em 10 episódios determinísticos.
- **Teste Cego Final:** O modelo final treinado foi submetido a **100 episódios de teste por semente** ($5 \times 100 = 500$ episódios por configuração), eliminando qualquer viés de seleção por pico momentâneo.
- **Métricas Internas da Rede Neural:** Logger customizado em tempo real para extrair a **Entropia da Política** (`policy_entropy`) e a **Divergência KL Aproximada** (`approx_kl`) durante cada atualização do gradiente.

---

## 📊 Resultados Experimentais

### Desempenho no Teste Cego Final (Média $\pm$ Desvio-Padrão para $n=5$)

| Configuração Experimental | Retorno Médio (100 ep.) | Taxa de Sucesso (%) | Tempo de Execução (min) |
| :--- | :---: | :---: | :---: |
| **DQN R1** (Baseline, 1 env) | $-106{,}7 \pm 3{,}2$ | $0{,}0\%$ | 21,1 min |
| **PPO R1** (Baseline, 1 env) | $+108{,}8 \pm 58{,}4$ | $25{,}3\%$ | 15,4 min |
| **DQN R2** (Sintonizado, 1 env) | **$+206{,}4 \pm 34{,}3$** | **$62{,}3\%$** | 28,4 min |
| **PPO R2** (Sintonizado, 16 envs) | $-165{,}3 \pm 1{,}1$ | $0{,}0\%$ | **7,5 min** |
| **PPO R3** (Ablação de Controle, 16 envs) | $+50{,}3 \pm 71{,}2$ | $12{,}5\%$ | 14,1 min |

### Validação Inferencial: Testes $t$ de Welch ($n=5$)
*Comparações com correção para variâncias desiguais:*

| Comparação Analisada | Estatística $t$ | $p$-valor | $d$ de Cohen | Intervalo de Confiança (95%) |
| :--- | :---: | :---: | :---: | :---: |
| **R1: DQN vs. PPO** | $-6{,}38$ | $0{,}0234$ | $-5{,}21$ | $[-360{,}8;\; -70{,}2]$ |
| **R2: DQN vs. PPO** | $+18{,}76$ | $0{,}0028$ | $+15{,}32$ | $[+286{,}5;\; +456{,}9]$ |
| **DQN: Baseline (R1) vs. Sintonizado (R2)** | $-15{,}74$ | $0{,}0037$ | $-12{,}85$ | $[-398{,}7;\; -227{,}5]$ |
| **PPO: Baseline (R1) vs. Sintonizado (R2)** | $+8{,}12$ | $0{,}0148$ | $+6{,}63$ | $[+129{,}1;\; +419{,}1]$ |

---

## 💡 Discussão Aprofundada e Achados Científicos

### 1. O Paradoxo do Ajuste (*The Tuning Paradox*)
Os dados evidenciam uma disparidade marcante entre os métodos:
- O **DQN** sob parâmetros padrão (R1) é completamente disfuncional (retorno negativo persistente de $-106{,}7$). A sintonia da R2 resgatou o algoritmo da inércia, elevando-o à convergência plena e ultrapassando a marca de resolução com **$+206{,}4$ pontos** e **62,3% de sucesso**.
- O **PPO**, por outro lado, apresentou comportamento diametralmente oposto: obteve aprendizado estável no baseline padrão (R1, $+108{,}8$), mas sofreu **degradação severa na R2 ($-165{,}3$, colapso absoluto)**.
*Conclusão:* Configurações rotuladas como "ótimas" em benchmarks não podem ser transpostas sem considerar as restrições orçamentárias de amostragem.

### 2. O Raio-X do Colapso: Diagnóstico via Ablação (R3)
Qual foi a causa real da falha do PPO na R2? Seria a taxa de aprendizado ou o fator de desconto ($\gamma=0{,}999$)?  
O grupo de controle de ablação (**R3**) solucionou a questão: ao manter todos os parâmetros idênticos à R1 e modificar **apenas** o número de ambientes paralelos ($n\_envs = 16$), o PPO voltou a apresentar retornos erráticos e colapsos periódicos.

A causa mecânica reside na **"Fome de Atualizações" (*Update Starvation*)**:
1. No PPO, os gradientes só são retropropagados após a coleta de um *rollout* de $n\_steps \times n\_envs$ amostras. Com $1024 \times 16$, cada ciclo exige **16.384 passos de ambiente**.
2. Sob um teto de 300.000 passos, o algoritmo realiza apenas cerca de **18 atualizações de pesos** em todo o treinamento!
3. O monitoramento das métricas internas (`latex/figs/raio_x.png`) comprovou que a **Entropia da Política despencou prematuramente** (de $\sim -1{,}38$ para $-1{,}13$), enquanto a **Divergência KL oscilou com picos abruptos** ($> 0{,}016$). Sem atualizações frequentes para refinar a política, a rede encerrou a exploração antes de aprender a navegar, consolidando uma política degenerada (*Policy Collapse*).

### 3. O Trade-Off: Eficiência de Amostras vs. Wall-Clock Time
- **DQN (R2):** Campeão absoluto em eficiência de dados virtuais. Com apenas 300.000 transições no buffer, resolve a tarefa complexa. Porém, por operar em processo único (*single-thread*), demandou **28,4 minutos**.
- **PPO (R2):** Campeão de velocidade física de execução, consumindo o orçamento de 300 mil passos em apenas **7,5 minutos** graças aos 16 processos paralelos. No entanto, sua estabilidade matemática requer um volume global de passos muito superior para permitir rollouts suficientes.

---

## 🖼️ Figuras e Gráficos do Estudo

- `latex/figs/curvas_avaliacao.png`: Curvas dinâmicas de evolução ao longo dos 300k passos na Rodada 1 (*Baseline*).
- `latex/figs/raio_x.png`: Gráfico bidirecional de Raio-X ilustrando o colapso do PPO R2 através da queda da Entropia da Política e oscilação da Divergência KL.
- `latex/figs/fig_curvas.png` e `latex/figs/fig_final.png`: Consolidação comparativa das avaliações determinísticas finais.

---

## 🛠️ Estrutura da Pasta e Compilação

```bash
03-dqn-ppo-lunarlander-v3/
├── README.md                          # Este documento detalhado
├── artigo_dqn_ppo_lunarlander.pdf     # Documento científico final em PDF
└── latex/
    ├── artigo.tex                     # Código-fonte LaTeX IEEEtran
    └── figs/
        ├── curvas_avaliacao.png       # Curvas de treino R1
        ├── raio_x.png                 # Raio-X do colapso de entropia/KL
        ├── fig_curvas.png             # Evolução consolidada
        ├── fig_curvas.pdf             # Vetorial da evolução
        ├── fig_final.png              # Boxplot final de teste
        └── fig_final.pdf              # Vetorial do teste final
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
@inproceedings{silvaneto2026lunarlander,
  author    = {Silva Neto, Jos{\'e} Joaquim da and Gomes, Pedro Henrique},
  title     = {Efici{\^e}ncia Amostral e Estabilidade de DQN e PPO em Navega{\c{c}}{\~a}o Aut{\^o}noma: Um Estudo de Abla{\c{c}}{\~a}o no LunarLander-v3},
  booktitle = {Anais da Disciplina de Intelig{\^e}ncia Artificial 2 - UnB FCTE},
  year      = {2026},
  address   = {Bras{\'i}lia, Brasil}
}
```
