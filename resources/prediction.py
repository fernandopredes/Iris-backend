from flask import request
from flask.views import MethodView
from flask_smorest import Blueprint, abort
from sqlalchemy.exc import SQLAlchemyError
from ml_model.model import predict_iris


from db import db
from models import PredictionModel
from schemas import PredictionSchema

blp = Blueprint("Predictions", __name__, description="Operações para criar e visualizar previsões")

@blp.route('/prediction')
class PredictionList(MethodView):
    @blp.arguments(PredictionSchema)
    @blp.response(201, PredictionSchema, description="Sucesso. Retorna as informações da previsão criada.")
    def post(self, prediction_data):
        # Extrai os recursos (features) dos dados de entrada
        features = [prediction_data['sepal_length'], prediction_data['sepal_width'],
                    prediction_data['petal_length'], prediction_data['petal_width']]

        # Chama a função predict_iris do modelo para fazer a previsão com base nos recursos
        predicted_class = predict_iris(features)

        # Cria um novo registro de previsão com os dados de entrada e a classe prevista
        prediction_record = PredictionModel(**prediction_data, predicted_class=predicted_class)

        try:
            db.session.add(prediction_record)
            db.session.commit()
        except SQLAlchemyError:
            abort(500, message="Um erro ocorreu ao tentar criar uma previsão.")

        return prediction_record

@blp.route('/predictions')
class Predictions(MethodView):
    @blp.response(200, PredictionSchema(many=True), description="Sucesso. Retorna a lista de previsões.")
    def get(self):
        """Rota para obter todas as previsões

        Retorna uma lista de todas as previsões realizadas.

        """
        predictions = PredictionModel.query.all()
        return predictions
