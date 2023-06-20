import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch

from torch.nn import functional as F

model_urls = {
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}

cfg = {
    'C': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
    'F': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512]
}


class VGG(nn.Module):
    def __init__(self, down=8, o_cn=1, final='abs', pretrained=False):
        super(VGG, self).__init__()
        self.down = down
        self.final = final
        self.features = make_layers(cfg['E'], batch_norm=False)
        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, o_cn, 1)
        )
        self._initialize_weights()
        self.features_list = []

        if pretrained:
            self.load_state_dict(model_zoo.load_url(model_urls['vgg19']), strict=False)
            print('Pretrained VGG19 loaded.')

    def forward(self, x):
        self.features_list = []
        x = self.features(x)
        if self.down < 16:
            x = F.interpolate(x, scale_factor=2)
        x = self.reg_layer(x)
        if self.final == 'abs':
            x = torch.abs(x)
        elif self.final == 'relu':
            x = torch.relu(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def regist_hook(self):
        self.features_list = []

        def get(model, input, output):
            # function will be automatically called each time, since the hook is injected
            self.features_list.append(output.detach())

        for name, module in self._modules['features']._modules.items():
            if name in ['1', '4', '9', '18', '27']:
                self._modules['features']._modules[name].register_forward_hook(get)
        for name, module in self._modules['reg_layer']._modules.items():
            if name in ['1']:
                self._modules['reg_layer']._modules[name].register_forward_hook(get)


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    # in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
