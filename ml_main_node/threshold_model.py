from torch import nn
import torch.nn.init as init


def init_weights(m):
    if isinstance(m, nn.Conv2d):  # Инициализация для сверточных слоев
        init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.Linear):  # Инициализация для полносвязных слоев
        init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif isinstance(m, nn.BatchNorm2d):  # Инициализация для слоев пакетной нормализации
        m.weight.data.fill_(1)
        m.bias.data.fill_(0)

def prepare_threshold_model(model):
   model.apply(init_weights)