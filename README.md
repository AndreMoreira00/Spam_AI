# Spam_AI

Este projeto é uma implementação de um classificador de spam usando técnicas de Aprendizado de Máquina. Ele utiliza um conjunto de dados de mensagens de texto rotuladas como spam ou não spam para treinar um modelo e prever se uma nova mensagem é spam ou não.

## Bibliotecas Utilizadas

- [numpy](https://numpy.org/): NumPy é uma biblioteca para a linguagem de programação Python, que suporta arrays e matrizes multidimensionais, além de possuir uma grande coleção de funções matemáticas de alto nível para operar com essas estruturas.

- [pandas](https://pandas.pydata.org/): Pandas é uma biblioteca de software escrita para a linguagem de programação Python para manipulação e análise de dados. Ela oferece estruturas de dados e operações para manipular tabelas numéricas e séries temporais.

- [scikit-learn](https://scikit-learn.org/stable/): Scikit-learn é uma biblioteca de aprendizado de máquina de código aberto que oferece ferramentas simples e eficientes para análise preditiva de dados, incluindo classificação, regressão, clustering, redução de dimensionalidade e seleção de modelos.

- [nltk](https://www.nltk.org/): NLTK (Natural Language Toolkit) é uma plataforma líder para a construção de programas Python para trabalhar com dados de linguagem humana.

## Funções Utilizadas

### `clean_text(text)`
Esta função realiza pré-processamento nos textos das mensagens, removendo pontuações, caracteres especiais e convertendo todas as palavras para letras minúsculas.

### `train_model(X_train, y_train)`
Esta função treina um modelo de classificação usando o conjunto de dados de treinamento fornecido (X_train, y_train). Neste caso, um modelo de classificação Naive Bayes é utilizado.

### `predict(model, X_test)`
Esta função utiliza o modelo treinado para prever se as mensagens no conjunto de dados de teste (X_test) são spam ou não spam.

### `evaluate_model(y_true, y_pred)`
Esta função avalia o desempenho do modelo comparando as previsões feitas pelo modelo (y_pred) com os rótulos reais (y_true), utilizando métricas como precisão, recall e pontuação F1.

## Detalhes do Projeto

Este projeto visa criar um classificador de spam eficaz usando técnicas de Aprendizado de Máquina. Ele explora a limpeza de texto, vetorização de palavras e modelos de classificação para alcançar seu objetivo.

## Exemplo de Uso

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils import clean_text, train_model, predict, evaluate_model

# Carregar o conjunto de dados
data = pd.read_csv('spam.csv')

# Pré-processamento dos dados
data['clean_text'] = data['text'].apply(clean_text)

# Dividir os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(data['clean_text'], data['spam'], test_size=0.2, random_state=42)

# Vetorização dos textos
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Treinar o modelo
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Fazer previsões
predictions = predict(model, X_test_tfidf)

# Avaliar o modelo
evaluate_model(y_test, predictions)
