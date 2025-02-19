# This files contain model definition for CNN classification networks
import torch
import torchvision.models as tvmodels
from torchvision import models
import torch.nn.functional as F
import torch.nn as nn
# from coral_pytorch.layers import CoralLayer
# from .gp_layer import *
# from .clip import *
from .resnet_pytorch import ResNet, Bottleneck
# from segment_anything import SamPredictor, sam_model_registry
# import clip
# from efficientnet_pytorch import EfficientNet


class resnet18(nn.Module):
    def __init__(self, num_classes, basemodel=tvmodels.resnet18, pretrained=True):
        super(resnet18, self).__init__()
        self.basemodel = basemodel(pretrained=pretrained)
        self.basemodel.fc = nn.Linear(
            in_features=2048, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.basemodel(x)
        return x


#trying to combine all resnet-50 modules, with a parameter for the changes in the last layer
class resnet50(nn.Module):
    def __init__(self, num_classes, basemodel=tvmodels.resnet50, pretrained=True):
        super(resnet50, self).__init__()
        self.basemodel = basemodel(pretrained=pretrained)
        self.basemodel.fc = nn.Linear(
            in_features=2048, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.basemodel(x)
        return x

class resnet50_regression(nn.Module):
    def __init__(self, num_classes, basemodel=tvmodels.resnet50, pretrained=True):
        super(resnet50_regression, self).__init__()
        self.basemodel = basemodel(pretrained=pretrained)
        self.basemodel.fc = nn.Linear(
            in_features=2048, out_features=1, bias=True)

    def forward(self, x):
        x = self.basemodel(x)
        return x

class resnet50_regression_features(nn.Module):
    def __init__(self, num_classes, basemodel=tvmodels.resnet50, pretrained=True):
        super(resnet50_regression_features, self).__init__()
        self.basemodel = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=1)      
        # self.fc = nn.Linear(
        #     in_features=2048, out_features=1, bias=True)

    def forward(self, x):
        outputs, features = self.basemodel(x)
        # outputs = self.fc(features)
        return (outputs, features)

# class resnet50_regression_rffgp(nn.Module):
#     def __init__(self, num_classes, basemodel=tvmodels.resnet50, pretrained=True):
#         super(resnet50_regression_rffgp, self).__init__()
#         self.basemodel = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=1)      
#         self.gp_layer = nn.RandomFourierFeaturesGP(input_dim=2048, output_dim=num_classes, hidden_dim = 100)

#     def forward(self, x):
#         _, features = self.basemodel(x)
#         outputs = self.gp_layer(features)
#         # outputs = self.fc(features)
#         return outputs

# class resnet50_coral(nn.Module):
#     def __init__(self, num_classes, basemodel=tvmodels.resnet50, pretrained=True):
#         super(resnet50_coral, self).__init__()
#         self.basemodel = basemodel(pretrained=pretrained)

#         self.basemodel.fc = CoralLayer(size_in=2048, num_classes=num_classes)

#     def forward(self, x):
#         x = self.basemodel(x)
#         return x

# class resnet50_corn(nn.Module):
#     def __init__(self, num_classes, basemodel=tvmodels.resnet50, pretrained=True):
#         super(resnet50_corn, self).__init__()
#         self.basemodel = basemodel(pretrained=pretrained)

#         self.basemodel.fc = nn.Linear(
#             in_features=2048, out_features=num_classes-1, bias=True)

#     def forward(self, x):
#         x = self.basemodel(x)
#         return x

# class cnn(nn.Module):
#     def __init__(self, num_classes, pretrained=False):
#         super().__init__()
#         self.conv1 = nn.Conv2d(256, 512, 32)
#         self.conv2 = nn.Conv2d(512, 1024, 16)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(1024, num_classes)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = F.relu(self.conv2(x))
#         x = torch.flatten(x, 1) # flatten all dimensions except batch
#         x = self.fc1(x)
#         return x

# class swin_transformer_tiny(nn.Module):
#     def __init__(self, num_classes, basemodel=tvmodels.swin_t, pretrained=True):
#         super(swin_transformer_base, self).__init__()
#         self.basemodel = basemodel(
#             weights=tvmodels.swin_transformer.Swin_T_Weights.DEFAULT)
#         self.basemodel.head = nn.Linear(
#             in_features=1024, out_features=num_classes)

#     def forward(self, x):
#         x = self.basemodel(x)
#         return x


# class swin_transformer_small(nn.Module):
#     def __init__(self, num_classes, basemodel=tvmodels.swin_s, pretrained=True):
#         super(swin_transformer_small, self).__init__()
#         self.basemodel = basemodel(
#             weights=tvmodels.swin_transformer.Swin_S_Weights.DEFAULT)
#         self.basemodel.head = nn.Linear(
#             in_features=768, out_features=num_classes)

#     def forward(self, x):
#         x = self.basemodel(x)
#         return x


# class swin_transformer_base(nn.Module):
#     def __init__(self, num_classes, basemodel=tvmodels.swin_b, pretrained=True):
#         super(swin_transformer_base, self).__init__()
#         self.basemodel = basemodel(
#             weights=tvmodels.swin_transformer.Swin_B_Weights.DEFAULT)
#         self.basemodel.head = nn.Linear(
#             in_features=1024, out_features=num_classes)

#     def forward(self, x):
#         x = self.basemodel(x)
#         return x

# class vision_transformer(nn.Module):
#     def __init__(self, num_classes, basemodel="vit_l_32", pretrained=True):
#         super(vision_transformer, self).__init__()
#         basemodels = {
#             "vit_b_16": (tvmodels.vit_b_16, tvmodels.ViT_B_16_Weights.DEFAULT),
#             "vit_b_32": (tvmodels.vit_b_32, tvmodels.ViT_B_32_Weights.DEFAULT),
#             "vit_l_16": (tvmodels.vit_l_16, tvmodels.ViT_L_16_Weights.DEFAULT),
#             "vit_l_32": (tvmodels.vit_l_32, tvmodels.ViT_L_32_Weights.DEFAULT),
#             "vit_h_14": (tvmodels.vit_h_14, tvmodels.ViT_H_14_Weights.DEFAULT),
#         }
#         self.basemodel, baseweights = basemodels[basemodel] if basemodel else basemodels["vit_b_16"]
#         if pretrained:
#             self.basemodel = self.basemodel(weights=baseweights)
#         self.basemodel.heads.head = nn.Linear(
#             in_features=self.basemodel.heads.head.in_features, out_features=num_classes
#         )
        
#     def forward(self, x):
#         x = self.basemodel(x)
#         return x

# class sam_regression(nn.Module):
#     def __init__(self, num_classes,  model_type = 'vit_b', train_encoder=False):
#         super(sam_regression, self).__init__()
#         checkpoint = {
#             'vit_h':'/scratchk/ehealth/optha/foundational_model_checkpoints/sam_vit_h_4b8939.pth',
#             'vit_l':'/scratchk/ehealth/optha/foundational_model_checkpoints/sam_vit_l_0b3195.pth',
#             'vit_b':'/scratchk/ehealth/optha/foundational_model_checkpoints/sam_vit_b_01ec64.pth'
#         }
#         self.sam_model = sam_model_registry[model_type](checkpoint=checkpoint[model_type])
#         self.classification_head = cnn(num_classes=1)

    
#     def forward(self, x):
#         x = self.sam_model.image_encoder(x)
#         return self.classification_head(x)

class efficientnet_v2_m_regression(nn.Module):
    def __init__(self, weights="DEFAULT", num_classes = 1, resolution = 512):
        super(efficientnet_v2_m_regression, self).__init__()
        self.basemodel = tvmodels.efficientnet_v2_m(weights = weights)
        self.basemodel.classifier = nn.Linear(
            in_features=1280, out_features=1, bias=True)
    
    def forward(self, x):
        x = self.basemodel(x)
        return x

class efficientnet_v2_m_regression_features(nn.Module):
    def __init__(self, weights="DEFAULT", num_classes = 1):
        super(efficientnet_v2_m_regression_features, self).__init__()
        self.basemodel = tvmodels.efficientnet_v2_m(weights = weights)
        self.basemodel.classifier = nn.Linear(
            in_features=1280, out_features=1, bias=True)
    
    def forward(self, x):
        x = self.basemodel.features(x)

        x = self.basemodel.avgpool(x)
        x = torch.flatten(x, 1)
        feat = x

        x = self.basemodel.classifier(x)

        return x , feat

class efficientnet_v2_m(nn.Module):
    def __init__(self, weights="DEFAULT", num_classes = 5, resolution = 512):
        super(efficientnet_v2_m, self).__init__()
        self.basemodel = tvmodels.efficientnet_v2_m(weights = weights)
        self.basemodel.classifier = nn.Linear(
            in_features=1280, out_features= num_classes, bias=True)
    
    def forward(self, x):
        x = self.basemodel(x)
        return x


class efficientnet_v2_m_sigmoid(nn.Module):
    def __init__(self, weights="DEFAULT", num_classes = 5):
        super(efficientnet_v2_m_sigmoid, self).__init__()
        self.basemodel = tvmodels.efficientnet_v2_m(weights = weights)
        self.basemodel.classifier = nn.Linear(
            in_features=1280, out_features= num_classes, bias=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.sigmoid(self.basemodel(x))
        return x

# class clip_resnet50_regression(nn.Module):
#     def __init__(self, num_classes, basemodel='RN50'):
#         super(clip_resnet50_regression, self).__init__()
#         model_parameters = {'name': basemodel}
#         self.basemodel = CLIP(model_parameters, adapt_avgpool=True)
#         self.fc = nn.Linear(
#             in_features=2048, out_features=1, bias=True)

#     def forward(self, x):
#         x = self.basemodel(x)
#         x = self.fc(x)
#         return x


# class gradability_model(nn.Module):
#     def __init__(self, num_classes = 3):
#         super(gradability_model, self).__init__()
#         self.model = EfficientNet.from_pretrained('efficientnet-b4')
#         self.model._fc = nn.Identity()
#         net_fl = nn.Sequential(
#                 nn.Linear(1792, 256),
#                 nn.ReLU(),
#                 nn.Dropout(p=0.5),
#                 nn.Linear(256, 64), 
#                 nn.ReLU(),
#                 nn.Dropout(p=0.5),
#                 nn.Linear(64, 3)
#                 )
#         self.model._fc = net_fl

#     def forward(self, x):
#         x = self.model(x)
#         return x



