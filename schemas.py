from marshmallow import Schema, fields

class PredictionSchema(Schema):
    """
    Define a estrutura dos dados para uma previsão da flor de Íris.
    """
    id = fields.Integer(description="ID da previsão", dump_only=True)
    sepal_length = fields.Float(required=True, description="Comprimento da sépala")
    sepal_width = fields.Float(required=True, description="Largura da sépala")
    petal_length = fields.Float(required=True, description="Comprimento da pétala")
    petal_width = fields.Float(required=True, description="Largura da pétala")
    predicted_class = fields.String(description="Classe prevista para a flor de Íris", dump_only=True)

    class Meta:
        description = "Define a estrutura de dados para uma previsão da flor de Íris."
