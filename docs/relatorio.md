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
    1. [4.1. Verificação de valores faltantes e consistência](#41-verificação-de-valores-faltantes-e-consistência)
    2. [4.2. Codificação de variáveis categóricas](#42-codificação-de-variáveis-categóricas)
    3. [4.3. Normalização de atributos numéricos](#43-normalização-de-atributos-numéricos)
    4. [4.4. Separação em conjuntos de treino e teste](#44-separação-em-conjuntos-de-treino-e-teste)
5. [5. Modelagem e avaliação](#5-modelagem-e-avaliação)
    1. [5.1. Modelos de classificação utilizados](#51-modelos-de-classificação-utilizados)
    2. [5.2. Protocolo de treinamento e avaliação](#52-protocolo-de-treinamento-e-avaliação)
    3. [5.3. Resultados obtidos](#53-resultados-obtidos)
6. [6. Resultados e Discussão](#6-resultados-e-discussão)
   1. [6.1. Comparação global entre os modelos](#61-comparação-global-entre-os-modelos)
   2. [6.2. Desempenho por classe (`HeartDisease = 0` e `HeartDisease = 1`)](#62-desempenho-por-classe-heartdisease--0-e-heartdisease--1)
      1. [6.2.1. Classe 0 – pacientes sem doença cardíaca](#621-classe-0--pacientes-sem-doença-cardíaca)
      2. [6.2.2. Classe 1 – pacientes com doença cardíaca](#622-classe-1--pacientes-com-doença-cardíaca)
   3. [6.3. Melhor e pior modelo sob a ótica da aplicação](#63-melhor-e-pior-modelo-sob-a-ótica-da-aplicação)
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

Nesta seção são apresentadas a caracterização estatística e a distribuição das variáveis do Heart Failure Prediction
Dataset. O objetivo é descrever o perfil dos pacientes e entender o comportamento dos atributos numéricos e categóricos,
preparando o terreno para as etapas de pré-processamento e modelagem.

## 3.1. Tabelas

A Tabela 2 resume as principais estatísticas descritivas dos atributos numéricos do conjunto de dados, incluindo mínimo,
máximo, média, desvio padrão e quartis. Foram consideradas como variáveis numéricas: `Age`, `RestingBP`, `Cholesterol`,
`FastingBS`, `MaxHR` e `Oldpeak`. Essas estatísticas permitem ter uma visão geral das faixas de valores observadas, bem
como da dispersão dos dados em cada atributo.

**Tabela 2 – Estatísticas descritivas dos atributos numéricos do dataset**

| Atributo    | Contagem | Média               | Desvio Padrão      | Mínimo | 1º quartil (25%) | Mediana (50%) | 3º quartil (75%) | Máximo |
|-------------|----------|---------------------|--------------------|--------|------------------|---------------|------------------|--------|
| Age         | 918.0    | 53.510893246187365  | 9.43261650673201   | 28.0   | 47.0             | 54.0          | 60.0             | 77.0   |
| RestingBP   | 918.0    | 132.39651416122004  | 18.5141541199078   | 0.0    | 120.0            | 130.0         | 140.0            | 200.0  |
| Cholesterol | 918.0    | 198.7995642701525   | 109.38414455220348 | 0.0    | 173.25           | 223.0         | 267.0            | 603.0  |
| FastingBS   | 918.0    | 0.23311546840958605 | 0.423045624739303  | 0.0    | 0.0              | 0.0           | 0.0              | 1.0    |
| MaxHR       | 918.0    | 136.80936819172112  | 25.4603341382503   | 60.0   | 120.0            | 138.0         | 156.0            | 202.0  |
| Oldpeak     | 918.0    | 0.8873638344226579  | 1.0665701510493257 | -2.6   | 0.0              | 0.6           | 1.5              | 6.2    |

Além disso, foi elaborado um resumo das variáveis categóricas (`Sex`, `ChestPainType`, `RestingECG`, `ExerciseAngina` e
`ST_Slope`), contendo as frequências absolutas e relativas de cada categoria. Esse resumo é apresentado na Tabela 3. A
partir dessas frequências é possível observar, por exemplo, quais tipos de dor torácica são mais comuns na amostra, qual
a distribuição por sexo e quais padrões de traçado eletrocardiográfico e de inclinação do segmento ST aparecem com maior
frequência.

**Tabela 3 – Frequências absolutas e relativas de cada categoria**

| Atributo         | Categoria | Frequência | Percentual |
|------------------|-----------|------------|------------|
| `Sex`            | M         | 725        | 78.98      |
|                  | F         | 193        | 21.02      |
| `ChestPainType`  | ASY       | 496        | 54.03      |
|                  | NAP       | 203        | 22.11      |
|                  | ATA       | 173        | 18.85      |
|                  | TA        | 46         | 5.01       |
| `RestingECG`     | Normal    | 552        | 60.13      |
|                  | LVH       | 188        | 20.48      | 
|                  | ST        | 178        | 19.39      |
| `ExerciseAngina` | N         | 547        | 59.59      |
|                  | Y         | 371        | 40.41      |
| `ST_Slope`       | Flat      | 460        | 50.11      |
|                  | Up        | 395        | 43.03      |
|                  | Down      | 63         | 6.86       |

Por fim, a Tabela 4 apresenta a distribuição da variável alvo `HeartDisease`, mostrando o número de pacientes sem (`0`)
e com (`1`) doença cardíaca, bem como a proporção de cada classe em relação ao total de 918 instâncias. Essa informação
é importante para avaliar se o problema de classificação é balanceado ou se há predominância de uma das classes, o que
pode impactar a escolha de métricas e modelos na etapa de avaliação.

**Tabela 4 – Distribuição da variável alvo**

| Presença de doença cardíaca | Frequência | Percentual |
|-----------------------------|------------|------------|
| 0                           | 410        | 44.66      |
| 1                           | 508        | 55.34      |

## 3.2. Gráficos

Para complementar as tabelas, foram gerados gráficos que permitem visualizar de forma mais intuitiva a distribuição das
variáveis.

- Para os atributos numéricos `Age`, `RestingBP`, `Cholesterol` e `MaxHR` foram construídos **histogramas**, nos quais o
  eixo horizontal representa faixas de valores e o eixo vertical representa a frequência de pacientes em cada faixa.
  Esses histogramas ajudam a identificar se as distribuições são mais concentradas em determinadas regiões, se há
  assimetrias marcantes ou presença de possíveis outliers.

|               **Gráfico 1 – Histograma `Age`**               |
|:------------------------------------------------------------:|
|         ![Histograma Age](assets/histograma_age.png)         |
|            **Gráfico 2 – Histograma `RestingBP`**            |
|   ![Histograma RestingBP](assets/histograma_restingbp.png)   |
|           **Gráfico 3 – Histograma `Cholesterol`**           |
| ![Histograma Cholesterol](assets/histograma_cholesterol.png) |
|              **Gráfico 4 – Histograma `MaxHR`**              |
|       ![Histograma MaxHR](assets/histograma_maxhr.png)       |

- Para as variáveis categóricas `Sex`, `ChestPainType`, `ExerciseAngina` e `ST_Slope` foram gerados **gráficos de
  barras**, em que cada barra corresponde a uma categoria e sua altura representa a contagem de pacientes naquela
  categoria. Esses gráficos permitem visualizar rapidamente qual categoria é dominante em cada atributo, de forma
  complementar às frequências mostradas na Tabela 3.

|                   **Gráfico 5 – Gráfico de barras para `Sex`**                   |
|:--------------------------------------------------------------------------------:|
|            ![Gráfico de barras Sex](assets/grafico_de_barras_sex.png)            |
|              **Gráfico 6 – Gráfico de barras para `ChestPainType`**              |
|  ![Gráfico de barras ChestPainType](assets/grafico_de_barras_chestpaintype.png)  |
|             **Gráfico 7 – Gráfico de barras para `ExerciseAngina`**              |
| ![Gráfico de barras ExerciseAngina](assets/grafico_de_barras_exerciseangina.png) |
|                **Gráfico 8 – Gráfico de barras para `ST_Slope`**                 |
|       ![Gráfico de barras ST_Slope](assets/grafico_de_barras_st_slope.png)       |

- Também foi construído um **gráfico de barras** para a variável alvo `HeartDisease`, destacando o número de pacientes
  com e sem doença cardíaca. Esse gráfico torna evidente o grau de balanceamento entre as duas classes e será utilizado
  como referência nas discussões sobre desempenho dos modelos de classificação na Seção 6.

| **Gráfico 9 – Gráfico de barras para a variável alvo** |
|:------------------------------------------------------:|
|    ![img.png](assets/grafico_de_barras_target.png)     |

Os gráficos produzidos nesta etapa auxiliam na interpretação das tabelas e fornecem uma visão exploratória inicial das
relações entre as variáveis, ainda que de forma univariada. Relações mais complexas (por exemplo, entre atributos
clínicos e a presença de doença) serão retomadas na discussão dos resultados dos modelos.

## 3.3. Tabela de valores faltantes

Como parte da análise exploratória, foi construída uma tabela específica para verificar a presença de **valores
faltantes** em cada atributo do dataset. A Tabela 5 apresenta, para cada coluna, o número absoluto de valores ausentes e
a porcentagem correspondente em relação ao total de 918 instâncias.

**Tabela 5 – Tabela de valores faltantes**

| Atributo       | Faltantes | Percentual |
|----------------|-----------|------------|
| Age            | 0         | 0.0        |
| Sex            | 0         | 0.0        |
| ChestPainType  | 0         | 0.0        |
| RestingBP      | 0         | 0.0        |
| Cholesterol    | 0         | 0.0        |
| FastingBS      | 0         | 0.0        |
| RestingECG     | 0         | 0.0        |
| MaxHR          | 0         | 0.0        |
| ExerciseAngina | 0         | 0.0        |
| Oldpeak        | 0         | 0.0        |
| ST_Slope       | 0         | 0.0        |
| HeartDisease   | 0         | 0.0        |

A partir dessa análise, verificou-se que o arquivo disponibilizado não contém valores faltantes nas variáveis
consideradas, ou seja, todas as colunas apresentam 0 valores ausentes. Isso simplifica o pré-processamento, pois não há
necessidade de aplicar técnicas de imputação de dados (como preenchimento por média, mediana ou categoria mais
frequente) ou de exclusão de registros por falta de informação.

Ainda assim, a construção dessa tabela é importante para documentar de forma explícita a qualidade dos dados utilizados
e justificar as decisões tomadas nas etapas seguintes. Na Seção 4, parte desse diagnóstico é retomado para explicar por
que não foram adotadas estratégias específicas de tratamento de valores ausentes.

---

# 4. Pré-processamento

Nesta seção são descritas as etapas de pré-processamento aplicadas ao Heart Failure Prediction Dataset antes do
treinamento dos modelos de classificação. O objetivo é garantir que as variáveis estejam em um formato adequado para os
algoritmos utilizados, evitando problemas relacionados a tipos de dados, codificação de categorias e escala dos
atributos numéricos.

Foram realizadas três etapas principais:

1. Verificação de valores faltantes e consistência básica dos dados;
2. Codificação das variáveis categóricas em formato numérico;
3. Normalização dos atributos numéricos para os modelos sensíveis à escala.

Ao final, os dados foram divididos em conjuntos de treino e teste.

## 4.1. Verificação de valores faltantes e consistência

Com base na análise exploratória apresentada na Seção 3, foi construída uma tabela de valores faltantes por atributo,
indicando tanto a quantidade absoluta quanto o percentual de ausências em relação ao total de 918 instâncias. Nessa
verificação, observou-se que todas as colunas do dataset apresentam **zero valores faltantes**, de modo que não foi
necessário aplicar técnicas de imputação (como preenchimento por média, mediana ou categoria mais frequente) ou remoção
de registros incompletos.

Além dos valores faltantes, foi realizada uma inspeção básica dos intervalos dos atributos numéricos (`Age`,
`RestingBP`, `Cholesterol`, `FastingBS`, `MaxHR` e `Oldpeak`) por meio das estatísticas descritivas e dos histogramas.
Os valores observados estiveram em faixas plausíveis para um conjunto de dados clínicos, de forma que não foram
aplicadas transformações específicas para remoção de *outliers*. Para fins deste trabalho, optou-se por **manter todos
os registros**, documentando apenas as distribuições observadas.

Como o atributo alvo `HeartDisease` já é disponibilizado como variável binária (`0` para ausência e `1` para presença de
doença cardíaca), não houve necessidade de transformá-lo para outro formato (por exemplo, rótulos nominais), sendo
utilizado diretamente na etapa de modelagem.

## 4.2. Codificação de variáveis categóricas

O dataset contém tanto variáveis numéricas quanto categóricas. As variáveis categóricas são:

- `Sex`
- `ChestPainType`
- `RestingECG`
- `ExerciseAngina`
- `ST_Slope`

Como a maior parte dos algoritmos de aprendizado de máquina utilizados neste trabalho espera atributos numéricos como
entrada, foi necessário converter essas variáveis categóricas para uma representação numérica. Para isso, foi empregada
a técnica de codificação *one-hot* (*one-hot encoding*), gerando colunas binárias para cada categoria distinta.

A codificação foi realizada utilizando a função `get_dummies` da biblioteca `pandas`, com o parâmetro `drop_first=True`.
Esse parâmetro faz com que, para cada atributo categórico, uma das categorias seja tomada como referência e não seja
explicitamente codificada, evitando colinearidade perfeita entre as colunas (o que é desejável sobretudo para modelos
lineares, como a regressão logística). Em termos práticos:

- Cada atributo categórico original é substituído por um conjunto de colunas indicadoras (0 ou 1).
- Uma categoria de cada atributo é omitida como base (por exemplo, `Sex_M` passa a ser representado por uma coluna
  `Sex_M`, enquanto `Sex_F` fica implícito nos casos em que `Sex_M = 0`).

Após essa etapa, obteve-se um novo *dataframe* totalmente numérico, no qual todas as variáveis categóricas foram
substituídas por suas respectivas colunas one-hot. O atributo alvo `HeartDisease` foi mantido separado, sem sofrer
transformação.

## 4.3. Normalização de atributos numéricos

Alguns algoritmos de classificação são **sensíveis à escala** dos atributos, como regressão logística, k-vizinhos mais
próximos (*k-NN*) e máquinas de vetores de suporte (SVM). Para esses modelos, é recomendável que as variáveis numéricas
tenham escalas comparáveis, evitando que atributos com valores absolutos maiores dominem a função de distância ou a
função de custo.

Com esse objetivo, foi aplicada uma normalização aos seguintes atributos numéricos:

- `Age`
- `RestingBP`
- `Cholesterol`
- `FastingBS`
- `MaxHR`
- `Oldpeak`

A normalização foi realizada utilizando um **escalonamento padrão** (*standardization*), por meio da classe
`StandardScaler` da biblioteca `scikit-learn`. Esse método transforma cada atributo para ter média aproximada a zero
e desvio padrão aproximadamente igual a um, com base nas estatísticas calculadas apenas sobre o conjunto de treino. Em
seguida, o mesmo ajuste é aplicado ao conjunto de teste.

Foram, assim, definidos dois conjuntos de atributos:

- Um conjunto **sem normalização**, adequado principalmente para modelos baseados em árvores (como árvores de decisão e
  florestas aleatórias), que não dependem da escala dos atributos;
- Um conjunto **normalizado**, utilizado nos modelos sensíveis à escala (por exemplo, regressão logística e SVM).

Essa abordagem permite comparar o desempenho dos algoritmos sob diferentes estratégias de pré-processamento, conforme
exigido na avaliação.

## 4.4. Separação em conjuntos de treino e teste

Após a codificação das variáveis categóricas e a preparação das versões normalizada e não normalizada dos atributos
numéricos, os dados foram divididos em conjuntos de **treino** e **teste**. A separação foi realizada utilizando a
função `train_test_split` da biblioteca `scikit-learn`.

Foram adotadas as seguintes configurações:

- **Proporção de divisão:** 70% das instâncias para treino e 30% para teste;
- **Estratificação:** a divisão considerou a variável alvo `HeartDisease` para manter, aproximadamente, a mesma
  proporção de pacientes com e sem doença cardíaca em ambos os conjuntos;
- **Semente aleatória:** foi fixado um valor para o parâmetro `random_state`, de forma a garantir a reprodutibilidade
  dos experimentos.

O conjunto de treino é utilizado para ajustar os parâmetros dos modelos de aprendizado de máquina, enquanto o conjunto
de teste é reservado exclusivamente para avaliar o desempenho final de cada modelo nas métricas de interesse (acurácia,
precisão, recall, F1 e tempo de treinamento), conforme detalhado na Seção 5.

---

# 5. Modelagem e avaliação

Nesta seção são descritos os modelos de classificação utilizados, o protocolo de avaliação adotado e os resultados
quantitativos obtidos na tarefa de prever a presença de doença cardíaca (`HeartDisease`) a partir dos atributos clínicos
do Heart Failure Prediction Dataset.

## 5.1. Modelos de classificação utilizados

Foram selecionados três algoritmos clássicos de classificação supervisionada, com características complementares:

- **Regressão Logística**: Modelo linear amplamente utilizado em problemas de classificação binária. Estima a
  probabilidade de um exemplo pertencer à classe positiva por meio de uma combinação linear dos atributos, seguida de
  uma função logística. Neste trabalho, foi utilizada a implementação `LogisticRegression` da biblioteca `scikit-learn`,
  com `max_iter = 1000` e `random_state = 42`. Como esse modelo é sensível à escala dos atributos, foram utilizados os
  dados na versão **normalizada** (conjunto `X_train_scaled` / `X_test_scaled`).
- **Árvore de Decisão**: Modelo baseado em partições recursivas do espaço de atributos, construindo uma estrutura em
  árvore em que cada nó interno representa uma condição de decisão sobre um atributo, e cada folha representa uma classe
  prevista. Foi utilizada a implementação `DecisionTreeClassifier` com `random_state = 42`. Como árvores não dependem
  diretamente da escala dos atributos, este modelo foi treinado com a versão **não normalizada** dos dados (conjunto
  `X_train` / `X_test`).
- **Random Forest**: Modelo de *ensemble* baseado em uma coleção de árvores de decisão treinadas sobre subconjuntos
  amostrais e de atributos, cujas previsões são combinadas, em geral por votação da maioria. A combinação de múltiplas
  árvores tende a reduzir overfitting e melhorar a capacidade de generalização. Neste trabalho, foi utilizada a
  implementação `RandomForestClassifier` com `n_estimators = 200` e `random_state = 42`, também sobre a versão **não
  normalizada** dos dados.

A escolha desses três algoritmos permite comparar um modelo linear (Regressão Logística) com modelos baseados em
árvores, tanto individual (Árvore de Decisão) quanto em conjunto (Random Forest).

## 5.2. Protocolo de treinamento e avaliação

O conjunto de dados, após o pré-processamento descrito na Seção 4, foi dividido em conjuntos de **treino** e **teste**,
utilizando a função `train_test_split` da biblioteca `scikit-learn`, com os seguintes parâmetros:

- **Proporção de divisão:** 70% das instâncias para treino e 30% para teste (`test_size = 0.3`);
- **Estratificação:** a variável alvo `HeartDisease` foi utilizada no parâmetro `stratify`, garantindo que a proporção
  de pacientes com e sem doença cardíaca fosse aproximadamente mantida tanto no conjunto de treino quanto no de teste;
- **Semente aleatória:** foi definido `random_state = 42`, de forma a tornar a divisão reprodutível.

Para cada modelo, o processo foi o seguinte:

1. Seleção do conjunto de atributos apropriado:
    - Modelos sensíveis à escala (Regressão Logística) foram treinados com `X_train_scaled` e avaliados com
      `X_test_scaled`;
    - Modelos baseados em árvores (Árvore de Decisão e Random Forest) foram treinados e avaliados com `X_train` e
      `X_test`.
2. Medição do **tempo de treinamento**:  
   Foi utilizada a função `time.time()` antes e depois da chamada ao método `fit` de cada modelo, considerando como
   tempo de treinamento a diferença entre esses dois instantes, medida em segundos.
3. Avaliação no conjunto de teste:  
   Após o treinamento, cada modelo gerou previsões `y_pred` para o conjunto de teste, a partir das quais foram
   calculadas as seguintes métricas:
    - **Acurácia** (`accuracy_score`): proporção de instâncias corretamente classificadas;
    - **Precisão** (`precision_score`): fração de exemplos previstos como classe positiva (`HeartDisease = 1`) que de
      fato pertencem a essa classe;
    - **Recall** (`recall_score`): fração de exemplos da classe positiva que foram corretamente identificados como tal;
    - **F1-score** (`f1_score`): média harmônica entre precisão e recall, resumindo o desempenho na classe positiva em
      um único valor;
    - **Tempo de treinamento**, em segundos.

Além dessas métricas globais, foi calculado o `classification_report` de cada modelo, contendo precisão, recall e F1 por
classe (`0` e `1`). Esses relatórios serão retomados na Seção 6 para discutir o desempenho por classe e comparar o
melhor e o pior modelo.

## 5.3. Resultados obtidos

A Tabela 6 apresenta o resumo dos resultados quantitativos obtidos pelos três modelos estudados no conjunto de teste,
considerando a variável alvo `HeartDisease`. Os valores de acurácia, precisão, recall e F1 referem-se à classe
positiva (`HeartDisease = 1`) e foram calculados com as funções da biblioteca `scikit-learn`.

**Tabela 6 – Desempenho dos modelos de classificação no conjunto de teste**

| Modelo              | Acurácia | Precisão | Recall | F1    | Tempo de treino (s) |
|---------------------|----------|----------|--------|-------|---------------------|
| Regressão Logística | 0,884    | 0,876    | 0,922  | 0,898 | 0,005               |
| Árvore de Decisão   | 0,812    | 0,839    | 0,817  | 0,828 | 0,004               |
| Random Forest       | 0,895    | 0,897    | 0,915  | 0,906 | 0,272               |

Observa-se que:

- O modelo de **Random Forest** apresentou a **maior acurácia global** (aproximadamente 0,895) e o maior F1-score (≈
  0,906), indicando bom equilíbrio entre precisão e recall na classe positiva, ao custo de um tempo de treinamento mais
  elevado (cerca de 0,27 s em comparação com os demais modelos).
- A **Regressão Logística** obteve desempenho bastante próximo ao da Random Forest, com acurácia em torno de 0,884 e
  F1 ≈ 0,898. Destaca-se, porém, pelo **maior recall** (≈ 0,922) entre os três modelos, o que significa que,
  proporcionalmente, identificou um número ligeiramente maior de pacientes com doença cardíaca, ao custo de um pequeno
  aumento de falsos positivos. Seu tempo de treinamento foi o menor, da ordem de milissegundos.
- A **Árvore de Decisão** foi o modelo com **desempenho global inferior**, apresentando acurácia de aproximadamente
  0,812 e F1 ≈ 0,828. Embora ainda apresente resultados razoáveis, mostrou-se menos eficaz que os demais tanto em
  acurácia quanto em F1, o que é compatível com o fato de ser um modelo único, mais sujeito a overfitting e a variações
  na partição dos dados.

Na Seção 6, esses resultados serão analisados em maior detalhe, considerando também as métricas por classe (`0` e `1`)
fornecidas pelos relatórios de classificação de cada modelo, de forma a identificar o melhor e o pior desempenho tanto
de forma global quanto para cada valor do atributo `HeartDisease`.


---

# 6. Resultados e Discussão

Nesta seção são discutidos os resultados obtidos pelos três modelos de classificação considerados — Regressão Logística,
Árvore de Decisão e Random Forest — tanto do ponto de vista global (acurácia e F1) quanto em termos de desempenho por
classe (`HeartDisease = 0` e `HeartDisease = 1`), com base nos relatórios de classificação gerados na Seção 5.

## 6.1. Comparação global entre os modelos

A Tabela 6 ([Seção 5.3](#53-resultados-obtidos)) mostrou que os três modelos obtiveram resultados razoáveis no problema
de predição de doença cardíaca, com acurácia variando aproximadamente entre 0,81 e 0,89. De forma geral:

- **Random Forest** apresentou o **melhor desempenho global**, com:
    - Acurácia ≈ **0,895**
    - Precisão ≈ **0,897**
    - Recall ≈ **0,915**
    - F1 ≈ **0,906**
- **Regressão Logística** ficou próxima, com:
    - Acurácia ≈ **0,884**
    - Precisão ≈ **0,876**
    - Recall ≈ **0,922**
    - F1 ≈ **0,898**
- **Árvore de Decisão** apresentou o **pior desempenho entre os três**, com:
    - Acurácia ≈ **0,812**
    - Precisão ≈ **0,839**
    - Recall ≈ **0,817**
    - F1 ≈ **0,828**

Em termos práticos, isso significa que todos os modelos conseguem diferenciar pacientes com e sem doença cardíaca com
desempenho razoável, mas o modelo de Random Forest se destaca por combinar boa precisão e bom recall de forma
equilibrada, resultando no maior F1-score. A Regressão Logística, apesar de ligeiramente inferior em acurácia e F1,
apresenta um desempenho muito próximo ao da Random Forest, com a vantagem de ter tempo de treinamento bem menor.

A Árvore de Decisão, por sua vez, é o modelo mais simples e interpretável, mas também o mais sujeito a overfitting e à
variação em função da partição dos dados, o que se reflete na acurácia e F1 inferiores em relação aos outros dois.

## 6.2. Desempenho por classe (`HeartDisease = 0` e `HeartDisease = 1`)

Para analisar com mais detalhes o comportamento dos modelos, é útil olhar para as métricas **por classe** nos relatórios
de classificação.

### 6.2.1. Classe 0 – pacientes sem doença cardíaca

Os resultados para a classe `0` (sem doença cardíaca) foram:

- **Regressão Logística**
    - Precisão: **0,90**
    - Recall: **0,84**
    - F1: **0,87**
    - Suporte: 123 exemplos

- **Árvore de Decisão**
    - Precisão: **0,78**
    - Recall: **0,80**
    - F1: **0,79**

- **Random Forest**
    - Precisão: **0,89**
    - Recall: **0,87**
    - F1: **0,88**

A Random Forest apresentou o **melhor equilíbrio** entre precisão e recall para a classe 0, com F1 ≈ 0,88. A Regressão
Logística teve precisão ligeiramente maior (0,90), porém com recall menor (0,84), ou seja, é um pouco mais conservadora:
quando indica que um paciente não tem doença, tende a acertar, mas deixa passar um pouco mais de casos dessa classe. A
Árvore de Decisão foi claramente a pior entre as três, com F1 ≈ 0,79, indicando desempenho inferior na identificação
correta de pacientes sem doença cardíaca.

### 6.2.2. Classe 1 – pacientes com doença cardíaca

Para a classe `1` (com doença cardíaca), os resultados foram:

- **Regressão Logística**
    - Precisão: **0,88**
    - Recall: **0,92**
    - F1: **0,90**
    - Suporte: 153 exemplos

- **Árvore de Decisão**
    - Precisão: **0,84**
    - Recall: **0,82**
    - F1: **0,83**

- **Random Forest**
    - Precisão: **0,90**
    - Recall: **0,92**
    - F1: **0,91**

Aqui, tanto a **Random Forest** quanto a **Regressão Logística** se destacam, com **recall ≈ 0,92** em ambos os casos.
Ou seja, aproximadamente 92% dos pacientes com doença cardíaca presentes no conjunto de teste foram corretamente
identificados como classe 1 por esses dois modelos. A Random Forest leva ligeira vantagem em F1 (≈ 0,91 vs. ≈ 0,90),
devido a uma precisão um pouco maior (0,90 contra 0,88).

Do ponto de vista clínico, um recall alto na classe positiva é desejável, pois significa que o modelo consegue
identificar a maior parte dos pacientes com doença cardíaca, reduzindo o número de **falsos negativos** (casos de doença
não detectados pelo modelo). Nesse aspecto, Random Forest e Regressão Logística se mostram adequadas, enquanto a Árvore
de Decisão, com F1 ≈ 0,83, apresenta desempenho sensivelmente inferior.

## 6.3. Melhor e pior modelo sob a ótica da aplicação

Considerando o contexto do problema — predição de doença cardíaca — e as métricas analisadas, é possível sintetizar os
resultados da seguinte forma:

- **Melhor modelo (globalmente): Random Forest**  
  A Random Forest apresentou a maior acurácia e o maior F1 global, além de manter bom desempenho em ambas as classes.
  Para a classe 1 (pacientes com doença), combinou recall elevado com boa precisão, o que a torna um candidato sólido
  para aplicações em que se deseja maximizar a detecção de casos positivos sem aumentar excessivamente os falsos
  positivos.

- **Modelo com melhor custo-benefício: Regressão Logística**  
  A Regressão Logística ficou muito próxima da Random Forest em termos de acurácia e F1, com recall ligeiramente
  superior na classe positiva e tempo de treinamento bem menor. Além disso, é um modelo mais simples e interpretável, o
  que pode ser uma vantagem em contextos clínicos em que se deseja justificar decisões com base em pesos e coeficientes
  associados às variáveis.

- **Pior modelo (entre os três): Árvore de Decisão**  
  Embora obtenha desempenho aceitável, a Árvore de Decisão apresenta acurácia e F1 substancialmente menores em
  comparação aos outros dois modelos, tanto na classe 0 quanto na classe 1. Isso sugere que, para este dataset em
  particular, uma única árvore não é suficiente para capturar adequadamente os padrões dos dados, sendo superada pelos
  modelos linear (Regressão Logística) e de ensemble (Random Forest).

Em resumo, os resultados indicam que modelos relativamente simples, como a Regressão Logística, já são capazes de obter
desempenho competitivo na predição de doença cardíaca a partir dos 11 atributos clínicos do *Heart Failure Prediction
Dataset*. A Random Forest melhora um pouco esses resultados ao custo de maior complexidade e tempo de treinamento,
enquanto a Árvore de Decisão isolada se mostra menos adequada como modelo final para este problema.

Trabalhos futuros poderiam incluir a comparação com outros algoritmos (como SVM ou k-vizinhos mais próximos), ajustes
sistemáticos de hiperparâmetros (*hyperparameter tuning*) e a utilização de técnicas de explicabilidade de modelos, de
forma a investigar mais detalhadamente quais atributos contribuem de maneira mais relevante para a predição da presença
de doença cardíaca.

---

# 7. Conclusão

- Resumir qual modelo seria indicado para uso prático
- Limitações (tamanho da amostra, apenas dados tabulares...)
