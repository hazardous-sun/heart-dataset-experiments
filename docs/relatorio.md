# Índice

1. [Introdução](#1-introdução)
    1. [1.1. Contexto de doenças cardíacas](#11-contexto-de-doenças-cardíacas)
    2. [1.2. Motivação de prever riscos a partir de dados clínicos](#12-motivação-de-prever-riscos-a-partir-de-dados-clínicos)
2. [2. Descrição do Dataset](#2-descrição-do-dataset)
    1. [2.1. Atributo Alvo](#21-atributo-alvo)
    2. [2.2. Tabela com variáveis](#22-tabela-com-variáveis)
3. [3. Análise exploratória](#3-análise-exploratória)
    1. [3.1. Tabelas](#31-tabelas)
    2. [3.2. Gráficos](#32-gráficos)
    3. [3.3. Tabela de valores faltantes](#33-tabela-de-valores-faltantes)
4. [4. Pré-processamento](#4-pré-processamento)
5. [5. Modelagem e avaliação](#5-modelagem-e-avaliação)
6. [6. Resultados e Discussão](#6-resultados-e-discussão)
7. [7. Conclusão](#7-conclusão)

---

# 1. Introdução

## 1.1. Contexto de doenças cardíacas

As doenças cardiovasculares constituem um dos principais problemas de saúde pública, estando entre as principais causas
de morte em diversos países. Em especial, quadros de doença arterial coronariana e insuficiência cardíaca estão
associados a internações frequentes, limitações funcionais e alto custo para os sistemas de saúde.

Na prática clínica, a identificação de pacientes com maior risco de desenvolver ou já apresentar doença cardíaca é feita
a partir de um conjunto de **fatores de risco e achados clínicos**, como idade, sexo, pressão arterial, níveis de
colesterol, presença de alterações eletrocardiográficas, sintomas de dor torácica e alterações durante o esforço físico.
Muitos desses fatores aparecem de forma estruturada em prontuários eletrônicos e exames complementares, criando um
cenário propício para o uso de técnicas de ciência de dados.

Neste trabalho, o foco não é discutir todos os aspectos clínicos das doenças cardíacas, mas sim investigar se
informações **clínicas simples, obtidas rotineiramente**, são suficientes para **alimentar modelos de aprendizado de
máquina** capazes de **distinguir pacientes com e sem doença cardíaca**, usando um conjunto de dados público amplamente
utilizado em estudos de predição.

## 1.2. Motivação de prever riscos a partir de dados clínicos

Com o avanço da informatização em saúde, tornou-se comum registrar de forma digital variáveis como idade, pressão
arterial de repouso, colesterol sérico, resultados de eletrocardiograma, frequência cardíaca máxima em teste de esforço
e presença de sintomas, entre outras. Esses dados estruturados podem ser utilizados para treinar **modelos de
classificação supervisionada** que auxiliem na tomada de decisão clínica.

Modelos desse tipo podem ser aplicados, por exemplo, para:

- Apoiar a **triagem** em serviços de pronto atendimento, destacando pacientes com maior probabilidade de apresentar
  doenças cardíaca;
- Auxiliar na **estratificação de risco**, indicando indivíduos que podem exigir acompanhamento mais próximo;
- Servir como ferramenta de **apoio à decisão**, complementando, e não substituindo, o julgamento do profissional de
  saúde.

Neste trabalho, utilizou-se o **Heart Failure Prediction Dataset**, disponível na plataforma Kaggle, como base para a
construção e comparação de diferentes modelos de classificação. A partir desse conjunto de dados, foram realizadas:

- Uma **análise exploratória** das variáveis disponíveis (Seção 3);
- Etapas de **pré-processamento** e codificação de atributos (Seção 4);
- Treinamento e avaliação de **modelos de classificação supervisionada** com métricas como acurácia, precisão, recall,
  F1 e tempo de treinamento (Seção 5);
- Uma **discussão dos resultados** obtidos, destacando o melhor e o pior desempenho entre os modelos considerados (Seção
  6).

---

# 2. Descrição do Dataset

O conjunto de dados utilizado neste trabalho é o **Heart Failure Prediction Dataset**, disponibilizado por F. Sorianο na
plataforma Kaggle. Ele combina informações clínicas de pacientes provenientes de diferentes bases de dados de doença
cardíaca, resultando em **918 instâncias** e **12 atributos**, sendo 11 atributos preditores e 1 atributo alvo (
`HeartDisease`).

Cada linha do arquivo representa um paciente, e cada coluna corresponde a uma variável clínica ou demográfica, como
idade, sexo, tipo de dor torácica, pressão arterial de repouso, colesterol sérico, presença de glicemia de jejum
elevada, alterações em eletrocardiograma de repouso, frequência cardíaca máxima atingida durante o exercício, presença
de angina induzida por esforço, depressão do segmento ST e inclinação do segmento ST, além do indicador binário de
presença de doença cardíaca.

O dataset está organizado em um único arquivo CSV, no qual não há valores ausentes nas colunas disponíveis, o que
simplifica as etapas iniciais de preparação dos dados. A Seção 3 detalha estatísticas descritivas das variáveis (tabelas
e gráficos), enquanto a Seção 4 descreve o pré-processamento aplicado antes do treinamento dos modelos.

- **Nome do dataset:** *Heart Failure Prediction Dataset*
- **Fonte:** Kaggle – F. Sorianο
- **Link:** https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction

## 2.1. Atributo Alvo

O atributo alvo deste estudo é `HeartDisease`, disponibilizado no dataset como uma variável **binária**:

- `0` – paciente **sem doença cardíaca**;
- `1` – paciente **com doença cardíaca**.

Do ponto de vista de modelagem, trata-se de um problema de **classificação binária**, em que o objetivo é aprender uma
função que, a partir dos 11 atributos preditores, atribua corretamente cada paciente à classe “com doença” ou “sem
doença”. Na prática, um modelo com bom desempenho nesse contexto pode contribuir para a identificação precoce de
pacientes com maior probabilidade de apresentar doença cardíaca, justificando investigações adicionais ou monitoramento
mais próximo.

Na Seção 5, diferentes algoritmos de classificação serão treinados para prever o valor de `HeartDisease`, e seus
desempenhos serão comparados com base em métricas de avaliação padrão em problemas de classificação.

## 2.2. Tabela com variáveis

A Tabela 1 apresenta os atributos disponíveis no Heart Failure Prediction Dataset, com o respectivo tipo e uma breve
descrição clínica resumida. As interpretações clínicas dessas variáveis serão utilizadas nas etapas de análise
exploratória (Seção 3) e na discussão dos resultados (Seção 6), especialmente para interpretar quais características se
relacionam mais fortemente com a presença de doença cardíaca.

**Tabela 1 – Atributos do Heart Failure Prediction Dataset**

| Nome             | Tipo           | Descrição clínica resumida                                                                                                                      |
|------------------|----------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| `Age`            | Numérico       | Idade do paciente, em anos. Idades mais avançadas tendem a estar associadas a maior risco cardiovascular.                                       |
| `Sex`            | Categórico     | Sexo biológico do paciente (`M` para masculino, `F` para feminino).                                                                             |
| `ChestPainType`  | Categórico     | Tipo de dor torácica (por exemplo, típica anginosa, atípica, não anginosa, assintomática). Relaciona-se à probabilidade de isquemia miocárdica. |
| `RestingBP`      | Numérico       | Pressão arterial sistólica de repouso, em mmHg. Valores elevados indicam possível hipertensão.                                                  |
| `Cholesterol`    | Numérico       | Colesterol sérico em jejum, em mg/dL. Níveis elevados são fator de risco para aterosclerose.                                                    |
| `FastingBS`      | Numérico (0/1) | Indicador de glicemia de jejum elevada (> 120 mg/dL: `1`; caso contrário: `0`). Relaciona-se a alteração glicêmica/diabetes.                    |
| `RestingECG`     | Categórico     | Resultado do eletrocardiograma de repouso (por exemplo, normal, sobrecarga ventricular, alterações de ST-T).                                    |
| `MaxHR`          | Numérico       | Frequência cardíaca máxima alcançada durante esforço. Valores muito baixos podem sugerir limitação da capacidade funcional.                     |
| `ExerciseAngina` | Categórico     | Presença (`Y`) ou ausência (`N`) de angina induzida por exercício.                                                                              |
| `Oldpeak`        | Numérico       | Depressão do segmento ST induzida por exercício, em relação ao repouso. Associada à isquemia miocárdica.                                        |
| `ST_Slope`       | Categórico     | Inclinação do segmento ST no pico do exercício (ascendente, plana ou descendente), relacionada à gravidade de isquemia.                         |
| `HeartDisease`   | Numérico (0/1) | Indicador de presença (`1`) ou ausência (`0`) de doença cardíaca. **É o atributo alvo do estudo.**                                              |

Essas variáveis serão exploradas quantitativamente na Seção 3, por meio de tabelas e gráficos, e servirão como insumo
para as etapas de pré-processamento (Seção 4) e modelagem (Seção 5), nas quais serão aplicadas técnicas de codificação
de variáveis categóricas e normalização de atributos numéricos antes do treinamento dos modelos de classificação.

---

# 3. Análise exploratória

## 3.1. Tabelas

- `describe()` dos numéricos
- Frequências das categorias (`value_counts`)

## 3.2. Gráficos

- Histogramas de `Age`, `RestingBP`, `Cholesterol`, `MaxHR`
- Gráficos de barras de `Sex`, `ChestPainType`, `ExerciseAngina`, `ST_Slope`
- Distribuição da classe `HeartDisease` (gráfico de barras)

## 3.3. Tabela de valores faltantes

---

# 4. Pré-processamento

- Normalização / padronização dos atributos numéricos (se usar KNN/SVM)
- Codificação de categóricos
    - One-hot encoding para `ChestPainType`, `RestingECG`, `ST_Slope`
    - Codificação binária simples para `Sex`, `ExerciseAngina`
- Justificar cada decisão

---

# 5. Modelagem e avaliação

- Descrever os 3 algoritmos escolhidos
- Explicar como foi feita a divisão treino/teste (ou k-fold)
- Tabelas com:
    - Acurácia, precisão, recall, F1 e tempo de treino por modelo
    - (Opcional) ROC curves / matriz de confusão.

---

# 6. Resultados e Discussão

- Apontar melhor e pior modelo
- Discutir performance por classe (usando `classification_report`)

---

# 7. Conclusão

- Resumir qual modelo seria indicado para uso prático
- Limitações (tamanho da amostra, apenas dados tabulares...)
