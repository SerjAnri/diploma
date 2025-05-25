import asyncio
import os
import aiohttp
import requests
import torch
import torchvision as tv
import numpy as np
from aiohttp import ClientTimeout
from flask import Flask, make_response, request, Blueprint
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from DeviceDataLoader import DeviceDataLoader
from accuracy import validate
from models_averaging import models_avg_accumulate, ema_accumulate, fedavg_accumulate
from swagger.swagger_config import swagger_config
from threshold_model import prepare_threshold_model
from flasgger import Swagger, swag_from

app = Flask(__name__)
swagger = Swagger(app, config=swagger_config)

request_mapping = Blueprint('main', __name__, url_prefix='/api/v1')

with app.app_context():
    torch.set_default_device('cpu')
    torch.cuda.empty_cache()

    urls_for_training = [url + '/train' for url in os.environ.get('URLS_FOR_WORKERS').split(',')]
    urls_for_updating = [url + '/update_weights' for url in os.environ.get('URLS_FOR_WORKERS').split(',')]
    urls_for_saving = [url + '/models_results' for url in os.environ.get('URLS_FOR_WORKERS').split(',')]
    urls_for_metrics = [url + '/model_metrics' for url in os.environ.get('URLS_FOR_WORKERS').split(',')]
    urls_for_cleaning = [url + '/clean_model_results' for url in os.environ.get('URLS_FOR_WORKERS').split(',')]

    device = DeviceDataLoader.get_default_device()
    print(device)
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform = tv.transforms.Compose([tv.transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                            tv.transforms.RandomHorizontalFlip(),
                            tv.transforms.ToTensor(),
                            tv.transforms.Normalize(*stats, inplace=True)])

    val_ds = tv.datasets.CIFAR10(root=os.environ.get('EXTRACTED_DATASET_PATH'), train=False, download=False, transform=transform)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    models_dir = os.environ.get('MODELS_DIR')
    threshold_model = resnet18(pretrained=True)
    threshold_model.fc = nn.Linear(512, 10)
    threshold_model = threshold_model.to(device)
    prepare_threshold_model(threshold_model)
    threshold_model_accuracy = validate(threshold_model, val_loader, device, True)

    common_model = resnet18(pretrained=False)
    common_model.fc = nn.Linear(512, 10)
    common_model = common_model.to(device)

@request_mapping.route('/start_model', methods=['GET'])
@swag_from('./swagger/start_model.yaml')
async def start_model():
    k = len(urls_for_training) + 1
    kfolds = requests.get(f'{os.environ.get('URL_FOR_KFOLD')}/{k}').json()
    one_kfold_length = len(kfolds[0]['trainingFiles'])
    print(f'Kfolds len: {len(kfolds)}')
    kfolds_matrix = []
    for i in range(0, 2):
        kfolds_matrix.append(kfolds[-i:] + kfolds[:-i])
    print(f'Kfolds matrix shape: {np.array(kfolds_matrix).shape}')

    training_info = dict(zip(urls_for_training, kfolds_matrix))

    for fold in range(0, k):
        print('__________________')
        print(f'Started global step: {fold}')
        models_results = await fetch_all_training(training_info, fold)
        models_paths = list(map(lambda res: res[0], list(filter(lambda res: res[1] > threshold_model_accuracy, models_results))))
        models_accuracies = list(map(lambda res: res[1], models_results))
        print(models_paths)
        income_models = []
        for model in models_paths:
            temp = resnet18()
            temp.fc = nn.Linear(512, 10)
            temp = temp.to(device)
            temp.load_state_dict(torch.load(model, weights_only=False))
            income_models.append(temp)
        if fold != k - 1:
            fedavg_accumulate(income_models, one_kfold_length, len(val_ds.data))
            torch.save(income_models[0].state_dict(), f'{models_dir}\\common_model.pt')
            await fetch_all_updating(urls_for_updating, f'{os.environ.get('MODELS_DIR')}\\common_model.pt')
            print(f'Finished global step: {fold}')
        else:
            print("Finalize models training")
            for filename in os.listdir(models_dir):
                if 'worker' in filename:
                    os.remove(os.path.join(models_dir, filename))
            max_accuracy_model_index = models_accuracies.index(max(models_accuracies))
            response = requests.get(urls_for_saving[max_accuracy_model_index]).content
            with open(f'{models_dir}\\final_model.pt', 'wb') as file:
                file.write(response)
            response_metrics = requests.get(urls_for_metrics[max_accuracy_model_index]).json()
            print(f'Final model metrics: {response_metrics}')
            await clean_models(urls_for_cleaning)
            print('Cleaned all workers results')
            print(f'Finished processing models')

    return make_response('', 200)


async def async_post(session, url):
    async with session.get(url) as response:
        return await response.json()

async def async_post_training(session, url, data):
    async with session.post(url, json=data) as response:
        return await response.content.read(), float(response.headers["accuracy"])

async def async_post_updating(session, url, file_path):
    with open(file_path, "rb") as file:
        async with session.post(url, data={'new_model.pt': file}) as response:
            return await response.json()

async def fetch_all_training(training_info, ind):
    async with aiohttp.ClientSession(timeout=ClientTimeout(total=10000000000)) as session:
        tasks = []
        for t in training_info.keys():
            tasks.append(async_post_training(session, t, training_info[t][ind]))
            print(f'Sent request to {t} with fold number {training_info[t][ind]['kFoldNumber']}')
        result = await asyncio.gather(*tasks)
        cnt = 1
        array_of_model_paths = []
        for res in result:
            with open(f'{models_dir}\\worker_{cnt}_fold{ind}_model.pt', 'wb') as file:
                file.write(res[0])
            array_of_model_paths.append((f'{models_dir}\\worker_{cnt}_fold{ind}_model.pt', res[1]))
            cnt += 1

        return array_of_model_paths

async def fetch_all_updating(urls, file_path):
    async with aiohttp.ClientSession(timeout=ClientTimeout(total=10000000000)) as session:
        tasks = []
        for url in urls:
            print(f'Send updating request to {url}')
            tasks.append(async_post_updating(session, url, file_path))
        return await asyncio.gather(*tasks)

async def clean_models(urls):
    async with aiohttp.ClientSession(timeout=ClientTimeout(total=10000000000)) as session:
        tasks = []
        for url in urls:
            tasks.append(async_post(session, url))
        return await asyncio.gather(*tasks)


app.register_blueprint(request_mapping)
if __name__ == '__main__':
    app.run()
