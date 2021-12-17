import torch
import torchvision.models as models
import requests, io
from torchvision import datasets, transforms as T
from PIL import Image

alexnet = models.alexnet(pretrained=True)

def alex(img_url):

    response=requests.get(img_url)

    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    img_pil = Image.open(io.BytesIO(response.content))        #image goes here
    img = transform(img_pil)

    feed = torch.unsqueeze(img, 0)
    out = alexnet(feed)

    with open('./imagenet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]
    
    _, indices = torch.sort(out, descending=True)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

    return [(classes[idx], percentage[idx].item()) for idx in indices[0][:2]]