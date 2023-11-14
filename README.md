# Iris Flower Prediction API 🌸

## Descrição
Este projeto é uma API Flask que realiza previsões sobre o tipo de flor de Íris (setosa, versicolor, virginica) com base em características da flor (comprimento e largura da sépala, comprimento e largura da pétala).

## Instalação 🛠️
Para instalar as dependências necessárias para o projeto, execute:

```bash
pip install -r requirements.txt
```


ChatGPT

Claro! Aqui está uma versão revisada do README.md com as instruções em um único bloco de código bash e com emojis para adicionar um toque mais amigável.

markdown

# Iris Flower Prediction API 🌸

## Descrição
Este projeto é uma API Flask que realiza previsões sobre o tipo de flor de Íris (setosa, versicolor, virginica) com base em características da flor (comprimento e largura da sépala, comprimento e largura da pétala).

## Instalação 🛠️
Para instalar as dependências necessárias para o projeto, execute:

```bash
pip install -r requirements.txt
```

## Uso 🚀

Para iniciar o servidor Flask, execute:

```bash
flask run
```
A API estará disponível em http://127.0.0.1:5000.

Fazendo Previsões

Para fazer uma previsão, envie uma solicitação POST para /prediction com os seguintes dados (substitua os valores de exemplo pelos seus dados):


```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

##Tecnologias Utilizadas 💻

  Python
    Flask
    Pandas
    scikit-learn
