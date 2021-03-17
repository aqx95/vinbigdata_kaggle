import timm
import torch.nn as nn


class Effnet(nn.Module):
    def __init__(self, model_name, n_class, drop_rate=0.0, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, drop_rate=drop_rate, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, n_class)

    def forward(self, x):
        x = self.model(x)
        return x

class Densenet(nn.Module):
    def __init__(self, model_name, n_class, drop_rate=0.0, pretrained=False):
      super().__init__()
      self.model = timm.create_model(model_name, drop_rate=drop_rate, pretrained=pretrained)
      n_features = self.model.num_features
      self.model.fc = nn.Linear(n_features, n_class)

    def forward(self, x):
        x = self.model(x)
        return x

#model factory
def create_model(config):
    if config.model == 'densenet':
        model_obj = Densenet
        model = Densenet(config.model_name, config.num_classes, config.drop_rate, config.pretrained)
        return model
    if config.model == "effnet":
        model_obj = Effnet
        model = Effnet(config.model_name, config.num_classes, config.drop_rate, config.pretrained)
        return model
