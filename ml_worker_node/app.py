import json
import os
import torch
import torchvision.transforms as tt
from flask import Flask, request, make_response, send_file, Blueprint
from torch import optim
from torchvision.models import resnet18
from DataLoaderService import DataLoaderService
from DeviceDataLoader import DeviceDataLoader
from swagger.swagger_config import swagger_config
from training import fit_one_cycle, validate
from flasgger import Swagger, swag_from


app = Flask(__name__)
swagger = Swagger(app, config=swagger_config)
request_mapping = Blueprint('worker', __name__, url_prefix='/api/v1')

with app.app_context():
    torch.set_default_device('cuda')
    torch.cuda.empty_cache()

    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_tfm = tt.Compose([tt.RandomCrop(32, padding=4, padding_mode='reflect'),
                            tt.RandomHorizontalFlip(),
                            tt.ToTensor(),
                            tt.Normalize(*stats, inplace=True)])
    val_tfm = tt.Compose([tt.ToTensor(), tt.Normalize(*stats)])

    extracted_dataset_path = os.environ.get('EXTRACTED_DATASET_PATH')

    device = DeviceDataLoader.get_default_device()
    val_dl = DataLoaderService(extracted_dataset_path).prepareValDataLoaders(val_tfm)
    val_dl = DeviceDataLoader(val_dl, device)
    print(f'Model device: {device} enabled')

    model = resnet18(pretrained=False)
    # models.fc = torch.nn.Linear(512, 100) #CIFAR100
    model.fc = torch.nn.Linear(512, 10) #CIFAR10

    epochs = int(os.environ.get('EPOCHS_NUMBER'))
    model_dir = os.environ.get('MODELS_DIR')

@request_mapping.route('/train', methods=['POST'])
@swag_from('./swagger/train_model.yaml')
def train():
    print('##########################')
    dto = request.json
    print(f'Started training for fold: {dto["kFoldNumber"]}')
    train_dl = DataLoaderService(extracted_dataset_path).prepareTrainDataLoaders(train_tfm, dto)
    train_dl = DeviceDataLoader(train_dl, device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.to(device)
    fit_one_cycle(epochs, model, train_dl, optimizer)
    del optimizer
    accuracy = validate(model, val_dl, device)['accuracy']
    torch.save(model.state_dict(), f'{model_dir}\\model.pt')
    response = send_file(f'{model_dir}\\model.pt', mimetype='multipart/form-data', as_attachment=True, download_name='model.pt')
    response.headers['accuracy'] = accuracy
    print(f'Finished training for fold: {dto["kFoldNumber"]}')
    return response

@request_mapping.route("/update_weights", methods=['POST'])
@swag_from('./swagger/update_weights.yaml')
def update_weights():
    print('__________________________')
    print('Started updating weights')
    file = request.files['new_model.pt']
    file.save(os.path.join(model_dir, 'new_model.pt'))
    model.load_state_dict(torch.load(f'{model_dir}\\new_model.pt', weights_only=False))
    result = make_response('{}', 200)
    result.mimetype = 'application/json'
    print('Finished updating weights')
    return result

@request_mapping.route('/clean_model_results', methods=['GET'])
@swag_from('./swagger/clean_model_results.yaml')
def clean_model_results():
    print('**************************')
    print('Started deleting models results')
    torch.cuda.empty_cache()
    for name, param in model.named_parameters():
        param.data.mul_(0)
    try:
        for filename in os.listdir(model_dir):
            if '.pt' in filename:
                os.remove(os.path.join(model_dir, filename))
        print('Finished deleting models results')
        response = make_response('{}', 200)
        response.mimetype = 'application/json'
        return response
    except Exception as e:
        exception = f'Failed to delete models results: {e}'
        print(exception)
        response = make_response({'exception': exception}, 500)
        response.mimetype = 'application/json'
        return response

@request_mapping.route('/model_results', methods=['GET'])
@swag_from('./swagger/get_model_results.yaml')
def get_model_results():
    response = send_file(f'{model_dir}\\model.pt', mimetype='multipart/form-data', as_attachment=True, download_name='model.pt')
    return response

@request_mapping.route('/model_metrics', methods=['GET'])
@swag_from('./swagger/get_model_metrics.yaml')
def get_model_metrics():
    accuracy = validate(model, val_dl, device)
    response = make_response(json.dumps(accuracy), 200)
    response.mimetype = 'application/json'
    return response

app.register_blueprint(request_mapping)
if __name__ == '__main__':
    app.run()
