import torch
import torchvision.models as models
import requests, io
from torchvision import transforms as T
from PIL import Image
from flask import jsonify

def detect(img_url, nnModel):
    
    if nnModel == "alexnet":
        model = models.alexnet(pretrained=True)
    elif nnModel == "resnet":
        model = models.resnet18(pretrained=True)
    elif nnModel == "squeezenet":
        model = models.squeezenet1_0(pretrained=True)
    elif nnModel == "vgg":
        model = models.vgg16(pretrained=True)
    elif nnModel == "densenet":
        model = models.densenet161(pretrained=True)
    elif nnModel == "inception":
        model = models.inception_v3(pretrained=True)
    elif nnModel == "googlenet":
        model = models.googlenet(pretrained=True)
    elif nnModel == "shufflenet":
        model = models.shufflenet_v2_x1_0(pretrained=True)
    elif nnModel == "mobilenet":
        model = models.mobilenet_v3_large(pretrained=True)
    elif nnModel == "resnext":
        model = models.resnext50_32x4d(pretrained=True)
    elif nnModel == "wide_resnet":
        model = models.wide_resnet50_2(pretrained=True)
    elif nnModel == "mnasnet":
        model = models.mnasnet1_0(pretrained=True)
    elif nnModel == "efficientnet":
        model = models.efficientnet_b6(pretrained=True)
    elif nnModel == "regnet_y":
        model = models.regnet_y_16gf(pretrained=True)
    elif nnModel == "regnet_x":
        model = models.regnet_x_16gf(pretrained=True)


    response=requests.get(img_url)

    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    img_pil = Image.open(io.BytesIO(response.content))        #image goes here
    img = transform(img_pil)

    feed = torch.unsqueeze(img, 0)
    out = model(feed)

    with open('./imagenet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]
    
    _, indices = torch.sort(out, descending=True)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

    return [(classes[idx], percentage[idx].item()) for idx in indices[0][:3]]