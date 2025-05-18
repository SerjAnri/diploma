from typing import List
import torch


def models_avg_accumulate(res_model: torch.nn.Module, folds_models: List[torch.nn.Module]):
    assert len(folds_models) > 0, "Empty training models list"
    for name, param in res_model.named_parameters():
        param.data.mul_(0)

    for fold_model in folds_models:
        model_par = dict(fold_model.named_parameters())
        for name, param in res_model.named_parameters():
            param.data.add_(model_par[name].data, alpha=1 / len(folds_models))

def ema_accumulate(folds_models: List[torch.nn.Module], decay: float = 0.999):
    init_model = folds_models[0]
    for name, param in init_model.named_parameters():
        for fold_model in folds_models[1:]:
            temp_dict = dict(fold_model.named_parameters())
            param.data.mul_(decay).add_(temp_dict[name].data, alpha=1 - decay)
