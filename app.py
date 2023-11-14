import os
from flask import Flask
from flask_smorest import Api
from flask_cors import CORS
from dotenv import load_dotenv
from ml_model.model import train_model
from db import db
from resources.prediction import blp as PredictionBlueprint

def create_app(db_url=None):
    app = Flask(__name__)
    load_dotenv()

    # Configurações básicas da API
    app.config["API_TITLE"] = "Iris Prediction API"
    app.config["API_VERSION"] = "v1"
    app.config["OPENAPI_VERSION"] = "3.0.3"
    app.config["OPENAPI_URL_PREFIX"] = "/"
    app.config["OPENAPI_SWAGGER_UI_PATH"] = "/swagger-ui"
    app.config["OPENAPI_SWAGGER_UI_URL"] = "https://cdn.jsdelivr.net/npm/swagger-ui-dist/"
    app.config["SQLALCHEMY_DATABASE_URI"] = db_url or os.getenv("DATABASE_URL", "sqlite:///data.db")
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["PROPAGATE_EXCEPTIONS"] = True

    # Inicialização do banco de dados
    db.init_app(app)

    # Configuração da API
    api = Api(app)

    # Enable CORS
    CORS(app)

    # Registro dos blueprints
    api.register_blueprint(PredictionBlueprint)

    # Criação das tabelas do banco de dados
    with app.app_context():
        db.create_all()

     # Treinar e salvar o modelo
    train_model()

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
