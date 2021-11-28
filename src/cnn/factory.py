import copy

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim
from torch.optim import lr_scheduler
import albumentations as A
import pretrainedmodels

from albumentations.pytorch import ToTensor

import ssl

from .dataset.custom_dataset import CustomDataset
from .transforms.transforms import RandomResizedCrop, RandomDicomNoise
from .utils.logger import log

class DensenetModel(nn.Module):
    ''' A densenet model. '''

    def __init__(self, name: str, n_output: int):
        super(DensenetModel, self).__init__()


        self.name = name
        self.n_output = n_output

        model_func = pretrainedmodels.__dict__[name]
        model = model_func(num_classes=1000, pretrained=None)
        model.load_state_dict(torch.load("./model/" + name))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        modules = list(model.children())[0]

        self.densenet = nn.Sequential(*modules)

        self.in_features = model.last_linear.in_features

        self.fc = nn.Linear(model.last_linear.in_features, n_output)

        self.relu = nn.ReLU()

    def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':

        x = self.relu(self.densenet(x))
        x = self.avg_pool(x).reshape(-1, self.in_features)
        x = self.fc(x)
        return x

def get_loss(cfg):
    #loss = getattr(nn, cfg.loss.name)(**cfg.loss.params)
    loss = getattr(nn, cfg.loss.name)(weight=torch.FloatTensor([2,1,1,1,1,1]).cuda(), **cfg.loss.params)
    log('loss: %s' % cfg.loss.name)
    return loss


def get_dataloader(cfg, folds=None):
    dataset = CustomDataset(cfg, folds)
    log('use default(random) sampler')
    loader = DataLoader(dataset, **cfg.loader)
    return loader


def get_transforms(cfg):
    def get_object(transform):
        if hasattr(A, transform.name):
            return getattr(A, transform.name)
        else:
            return eval(transform.name)
    transforms = [get_object(transform)(**transform.params) for transform in cfg.transforms]
    return A.Compose(transforms)


def get_model(cfg):
    # ssl._create_default_https_context = ssl._create_unverified_context
    #urllib.request.urlopen(urllink)

    log(f'model: {cfg.model.name}')
    log(f'pretrained: {cfg.model.pretrained}')

    if cfg.model.name.endswith('_wsl'):
        model = torch.hub.load('facebookresearch/WSL-Images', cfg.model.name)
        model.fc = torch.nn.Linear(2048, cfg.model.n_output)
        return model
    elif cfg.model.name.startswith('efficientnet'):
        from efficientnet_pytorch import EfficientNet
        model = EfficientNet.from_pretrained(cfg.model.name, num_classes=cfg.model.n_output)
        #model.set_swish(memory_efficient=False)
        #model._fc = torch.nn.Linear(1280, cfg.model.n_output)
        return model
    elif 'googlenet' in cfg.model.name:
        import torchvision.models as models
        model = models.googlenet(pretrained=False, num_classes=1000)
        model.load_state_dict(torch.load("./model/" + cfg.model.name))
        model.aux_logits = False
        model.aux1 = None  # type: ignore[assignment]
        model.aux2 = None  # type: ignore[assignment]
        #model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.fc = nn.Linear(1024, cfg.model.n_output)
        return model

    try:
        model_func = pretrainedmodels.__dict__[cfg.model.name]
    except KeyError as e:
        model_func = eval(cfg.model.name)

    if "densenet" in cfg.model.name:
        return DensenetModel(cfg.model.name, cfg.model.n_output)

    model = model_func(num_classes=1000, pretrained=None)
    model.load_state_dict(torch.load("./model/" + cfg.model.name))
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    model.last_linear = nn.Linear(
        model.last_linear.in_features,
        cfg.model.n_output,
    )
    return model


def get_optim(cfg, parameters):
    optim = getattr(torch.optim, cfg.optim.name)(parameters, **cfg.optim.params)
    log(f'optim: {cfg.optim.name}')
    return optim


def get_scheduler(cfg, optim, last_epoch):
    if cfg.scheduler.name == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optim,
            **cfg.scheduler.params,
        )
        scheduler.last_epoch = last_epoch
    else:
        scheduler = getattr(lr_scheduler, cfg.scheduler.name)(
            optim,
            last_epoch=last_epoch,
            **cfg.scheduler.params,
        )
    log(f'last_epoch: {last_epoch}')
    return scheduler

