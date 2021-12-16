# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 13:17:51 2021

@author: DELL
"""

from flask import Flask, render_template, request

import torch
import torchvision.models as models
import requests, io
from torchvision import datasets, transforms as T
from PIL import Image

alexnet = models.alexnet(pretrained=True)

app=Flask(__name__)

@app.route('/',methods=['GET'])

def hello():
    return render_template('home.html')

@app.route('/',methods=['POST'])


def predict():
    img_url=request.form.get('image')

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
    cls = [(classes[idx], percentage[idx].item()) for idx in indices[0][:3]]

    return render_template('home.html', pred=cls)

if __name__ == '__main__':
    app.run(port=3000,debug=True)