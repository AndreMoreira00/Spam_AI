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
import joblib
model_svm = joblib.load('model_svm.pkl')
X_treinamento = joblib.load('X_treinamento.pkl')
vet = joblib.load('vet.pkl')

text = str(input())
print(text)

from sklearn.feature_extraction.text import CountVectorizer
vt = CountVectorizer(ngram_range=(1, 2),vocabulary='vet', max_features=42858, )
text = vt.transform(text)
text

previsao = model_svm.predict(text)
previsao
```

## Licença
Este projeto é licenciado sob a MIT License.
