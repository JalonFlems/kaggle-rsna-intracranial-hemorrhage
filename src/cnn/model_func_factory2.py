import torch
import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels


class DensenetModel(nn.Module):
    ''' A densenet model. '''

    def __init__(self, name: str, n_outputs: int):
        super(DensenetModel, self).__init__()

        self.name = name


        model_func = pretrainedmodels.__dict__[name]
        model = model_func(num_classes=1000, pretrained=None)
        model.load_state_dict(torch.load("../../model/" + name))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        modules = list(model.children())[0]

        self.densenet = nn.Sequential(*modules)
        
        self.in_features = model.last_linear.in_features

        self.fc = nn.Linear(model.last_linear.in_features, n_outputs)

    def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':

        x = F.relu(self.densenet(x))
        x = self.avg_pool(x).reshape(-1, self.in_features)
        x = self.fc(x)
        return x



#model_func = pretrainedmodels.__dict__['densenet169']
#
#model = model_func(num_classes=1000, pretrained=None)
#model.load_state_dict(torch.load("../../model/densenet169"))
#model.avg_pool = nn.AdaptiveAvgPool2d(1)
#model.last_linear = nn.Linear(
#    model.last_linear.in_features,
#    6,
#)

print(pretrainedmodels.__dict__['densenet201'])

model = DensenetModel("densenet201", 6)

out = model(torch.randn(1, 3, 512, 512))

import pdb; pdb.set_trace()



#params = list(model.parameters())
#
#layers = list(model.children())
#
#x = torch.randn(1, 3, 512, 512)
#
#for layer in layers:
#    try:
#        x = layer(x)
#    except Exception:
#        import pdb; pdb.set_trace()
