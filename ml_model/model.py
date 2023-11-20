import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os

def train_model():
    # Carrega o conjunto de dados Iris
    iris = load_iris()
    dataset = pd.DataFrame(data=iris['data'], columns=iris['feature_names'])
    X = dataset  # Características (features) das flores.
    y = iris['target']  # Classes das flores.

    # Divide os dados em conjuntos de treino e teste.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

    # Define os parâmetros para otimização do modelo KNN.
    param_grid = {
        'n_neighbors': range(1, 31),
        'metric': ['euclidean', 'manhattan', 'minkowski'],
        'weights': ['uniform', 'distance']
    }

    # Usa GridSearchCV para encontrar os melhores hiperparâmetros para o modelo KNN.
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Seleciona o modelo com os melhores hiperparâmetros.
    best_model = grid_search.best_estimator_

    # Treina o melhor modelo com todo o conjunto de dados.
    best_model.fit(X, y)

    # Salva o modelo treinado para uso posterior.
    model_file_path = os.path.join(os.getcwd(), 'best_iris_model.pkl')
    joblib.dump(best_model, model_file_path)

def load_model():
    # Carrega o modelo treinado do arquivo.
    model_file_path = os.path.join(os.getcwd(), 'best_iris_model.pkl')
    return joblib.load(model_file_path)

def predict_iris(features):
    # Carrega o modelo e usar para fazer previsões com base em novos dados de entrada.
    model = load_model()
    iris = load_iris()

    # Cria um DataFrame com os nomes das colunas correspondentes.
    feature_names = iris['feature_names']
    input_data = pd.DataFrame([features], columns=feature_names)

    # Faz a previsão usando o DataFrame.
    prediction = model.predict(input_data)
    predicted_class = iris['target_names'][prediction][0]
    return predicted_class

if __name__ == '__main__':
    train_model()
    test_features = [5.1, 3.5, 1.4, 0.2]
    print(predict_iris(test_features))
