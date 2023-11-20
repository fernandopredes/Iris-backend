import pandas as pd
import pytest
from sklearn.metrics import accuracy_score
from ml_model.model import train_model, load_model, predict_iris
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

@pytest.fixture
def iris_dataset():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=7)
    return X_train, X_test, y_train, y_test

def test_model_accuracy(iris_dataset):
    X_train, X_test, y_train, y_test = iris_dataset
    iris = load_iris()
    feature_names = iris['feature_names']

    # Convertendo X_test para DataFrame com os nomes das colunas
    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    # Treinar e salvar o modelo
    train_model()

    # Carregar o modelo treinado
    model = load_model()

    # Fazer previsões no conjunto de teste
    predictions = model.predict(X_test_df)

    # Calcular a precisão
    accuracy = accuracy_score(y_test, predictions)

    # Verificar se a precisão atende ao threshold definido
    assert accuracy > 0.90
