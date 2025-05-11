import json

from flask import Flask
import configparser
from flask import make_response
from flasgger import Swagger, LazyString, swag_from
from service.KFoldDividerService import KFoldDividerService

config = configparser.ConfigParser()
config.read('config.ini')

app = Flask(__name__)

swagger_config = {
    "headers": [
    ],
    "specs": [
        {
            "endpoint": 'apispec_1',
            "route": '/apispec_1.json',
            "rule_filter": lambda rule: True,  # all in
            "model_filter": lambda tag: True,  # all in
        }
    ],
    "static_url_path": "/flasgger_static",
    # "static_folder": "static",  # must be set by user
    "swagger_ui": True,
    "specs_route": "/apidocs/",
    "info": {
        "title": "KFoldDivider API",
    }
}

swagger = Swagger(app, config=swagger_config)

@app.route('/api/v1/getKFolds', methods=['GET'])
@swag_from('spec.yaml')
def getKFolds():
    divider = KFoldDividerService(config['DEFAULT']['labelsPath'], int(config['DEFAULT']['k']))
    responseBody = divider.generateKFolds()
    for r in responseBody:
        print(len(r.trainFiles))
    response = make_response( json.dumps([obj.__dict__ for obj in responseBody]), 200)
    response.mimetype = 'application/json'
    return response


if __name__ == '__main__':
    app.run()
