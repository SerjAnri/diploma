import asyncio
from collections import defaultdict

import aiohttp
import requests
import torch
import numpy as np
from aiohttp import ClientTimeout
from flask import Flask, make_response, request


app = Flask(__name__)



async def async_post(session, url, data):
    async with session.post(url, json=data) as response:
        return await response.json()

async def fetch_all_training(training_info, ind):
    async with aiohttp.ClientSession(timeout=ClientTimeout(total=10000000000)) as session:
        tasks = []
        # print(ind)
        for t in training_info.keys():
            tasks.append(async_post(session, t, training_info[t][ind]))
            # print(training_info[t][ind]['kFoldNumber'])
        # print('_________________________________')
        return await asyncio.gather(*tasks)

async def fetch_all_updating(urls, training_info_result):
    async with aiohttp.ClientSession(timeout=ClientTimeout(total=10000000000)) as session:
        tasks = []
        for url in urls:
            tasks.append(async_post(session, url, training_info_result))
        return await asyncio.gather(*tasks)

@app.route('/start_model', methods=['GET'])
async def start_model():
    k = 2 # TODO перенести в конфиг
    kfolds = requests.get("http://127.0.0.1:5002/api/v1/getKFolds").json()
    kfolds_matrix = []
    for i in range(0, k):
        kfolds_matrix.append(kfolds[-i:] + kfolds[:-i])

    urls_for_training = ["http://127.0.0.1:5000/train", "http://127.0.0.1:5005/train"]
    urls_for_updating = ["http://127.0.0.1:5000/update_weights", "http://127.0.0.1:5005/update_weights"]


    training_info = dict(zip(urls_for_training, kfolds_matrix))

    for i in range(0, k):
        training_results = await fetch_all_training(training_info, i) # обучение модели и получение весов
        print(training_results)
        training_results = list(filter(lambda t: t['accuracy'] > 0.2, training_results)) #TODO порог вычислять на основе модели с шумом
        print('len of train-res after filtering', len(training_results))
        new_weights = defaultdict()
        for ti in training_results:
            temp = {key: torch.tensor(np.array(value), dtype=torch.float) for key, value in ti['state_dict'].items()}

            for key, value in temp.items():
                if new_weights.get(key) is not None:
                    new_weights[key] = torch.add(value * (1 / 2), new_weights[key])
                else:
                    new_weights[key] = value * (1 / 2)

        new_weights = {key: value.detach().cpu().numpy().tolist() for key, value in new_weights.items()}
        for ti in training_results:
            ti['state_dict'] = new_weights
        await fetch_all_updating(urls_for_updating, training_results[0])

    return make_response('', 200)

if __name__ == '__main__':
    app.run()
