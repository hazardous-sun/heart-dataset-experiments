# Predição de Doença Cardíaca com Machine Learning

Este projeto implementa um pipeline de *machine learning* para prever a presença de doença cardíaca em pacientes, a
partir de variáveis clínicas (idade, pressão arterial de repouso, colesterol, tipo de dor torácica, etc.).

O foco é comparar modelos clássicos de classificação (Regressão Logística, Árvore de Decisão e Random Forest) e discutir
seu uso em um contexto de apoio à decisão clínica.

> 📄 **Documentação completa:** consulte o arquivo [`docs/relatorio.md`](docs/relatorio.md) para uma descrição detalhada
> do problema, metodologia, resultados e conclusões.

---

## 🔍 Objetivo

- Prever a variável-alvo **`HeartDisease`** (1 = presença de doença, 0 = ausência)
- Comparar o desempenho de diferentes modelos de classificação
- Discutir métricas relevantes em contextos médicos (precisão, recall, F1, etc.)

---

## 🧬 Dados

- Arquivo principal: `data/heart.csv`
- Número de exemplos: ~918 pacientes
- Tipo de problema: **classificação binária**
- Atributos: variáveis demográficas e clínicas (ex.: idade, sexo, pressão arterial, colesterol, tipo de dor torácica).

Mais detalhes sobre o dataset estão em [`docs/relatorio.md`](docs/relatorio.md).

---

## 🧠 Modelos Utilizados

Os principais modelos treinados e avaliados são:

- **Regressão Logística**
- **Árvore de Decisão**
- **Random Forest**

A comparação entre eles é feita com base em:

- Acurácia
- Precisão
- Recall
- F1-score
- Tempo de treinamento

A análise completa dos resultados está em [`docs/relatorio.md`](docs/relatorio.md).
