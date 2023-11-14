import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os

def train_and_save_model():
    iris = load_iris()
    dataset = pd.DataFrame(data=iris['data'], columns=iris['feature_names'])
    X = dataset
    y = iris['target']

    # Dividindo os dados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

    # Treinando o modelo
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)

    # Salvando o modelo treinado
    model_file_path = os.path.join(os.getcwd(), 'iris_model.pkl')
    joblib.dump(model, model_file_path)

def load_model():
    model_file_path = os.path.join(os.getcwd(), 'iris_model.pkl')
    return joblib.load(model_file_path)

def predict_iris(features):
    model = load_model()
    iris = load_iris()
    prediction = model.predict([features])
    predicted_class = iris['target_names'][prediction][0]
    return predicted_class



if __name__ == '__main__':
    test_features = [5.1, 3.5, 1.4, 0.2]
    print(predict_iris(test_features))
