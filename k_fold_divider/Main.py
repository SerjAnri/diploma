import json
import os
from flask import Flask
from flask import make_response
from flasgger import Swagger, swag_from
from service.KFoldDividerService import KFoldDividerService
from swagger.swagger_config import swagger_config


app = Flask(__name__)
swagger = Swagger(app, config=swagger_config)

@app.route('/api/v1/getKFolds/<int:quantity>', methods=['GET'])
@swag_from('./swagger/spec.yaml')
def getKFolds(quantity):
    divider = KFoldDividerService(quantity)
    responseBody = divider.generateKFolds()
    response = make_response( json.dumps([obj.__dict__ for obj in responseBody]), 200)
    response.mimetype = 'application/json'
    return response


if __name__ == '__main__':
    app.run()
