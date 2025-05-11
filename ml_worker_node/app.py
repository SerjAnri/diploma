import json

import torch
import numpy as np
import torchvision.transforms as tt
from flask import Flask, request, make_response
from torchvision.models import resnet18

from DataLoaderService import DataLoaderService
from DeviceDataLoader import DeviceDataLoader, to_device
from JSONTransformer import convert_input_to, KFoldDto
from Training import fit_one_cycle, validate
from WorkerResponseDto import WorkerResponseDto

app = Flask(__name__)

with app.app_context():
    torch.set_default_device('cuda')
    torch.cuda.empty_cache()

    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_tfm = tt.Compose([tt.RandomCrop(32, padding=4, padding_mode='reflect'),
                            tt.RandomHorizontalFlip(),
                            tt.ToTensor(),
                            tt.Normalize(*stats, inplace=True)])
    val_tfm = tt.Compose([tt.ToTensor(), tt.Normalize(*stats)])
    extracted_train_path = 'C:\\Users\\Asus\\Desktop\\pythonProject\\data'
    extracted_test_path = 'C:\\Users\\Asus\\Desktop\\pythonProject\\data'
    device = DeviceDataLoader.get_default_device()
    val_dl = DataLoaderService(extracted_train_path, extracted_test_path).prepareValDataLoaders(val_tfm)
    val_dl = DeviceDataLoader(val_dl, device)

    # Загрузка предобученной модели
    model = resnet18(pretrained=False)

    # Заменяем последний полносвязный слой
    model.fc = torch.nn.Linear(model.fc.in_features, 100)

    epochs = 20 #TODO перенести в конфиг


@app.route('/train', methods=['POST'])
@convert_input_to(KFoldDto, request=request)
def train(dto):
    train_dl = DataLoaderService(extracted_train_path, extracted_test_path).prepareTrainDataLoaders(train_tfm, dto)
    train_dl = DeviceDataLoader(train_dl, device)

    fit_one_cycle(epochs, model, train_dl)
    accuracy = validate(model, val_dl)
    print(model.state_dict().keys())
    result = WorkerResponseDto(
        state_dict={key: value.detach().cpu().numpy().tolist() for key, value in model.state_dict().items()},
        accuracy=accuracy,
    )
    response = make_response(json.dumps(result.__dict__), 200)
    response.mimetype = 'application/json'
    return response

@app.route("/update_weights", methods=['POST'])
def update_weights():
    state_dict = json.loads(json.dumps(request.get_json()))['state_dict']
    tensor_state_dict = {key: torch.tensor(np.array(value), dtype=torch.float).cuda() for key, value in state_dict.items()}
    model.load_state_dict(tensor_state_dict, strict=True, assign=True)
    result =  make_response('{}', 200)
    result.mimetype = 'application/json'
    return result


if __name__ == '__main__':
    app.run()
