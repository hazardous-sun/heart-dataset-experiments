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

## 1.2. Motivação de prever riscos a partir de dados clínicos

---

# 2. Descrição do Dataset

- Nome e link do dataset
  - Heart Failure Prediction Dataset
  - https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction

## 2.1. Atributo Alvo

- Atributo alvo
  - `HeartDisease`

## 2.2. Tabela com variáveis

- Nome
- Tipo
- Descrição clínica resumida

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
