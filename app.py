# -*- coding: utf-8 -*-

from flask import Flask, render_template, request

import torch, torchvision, cv2, argparse
import numpy as np
import torchvision.models as models
import requests, io
from torchvision import datasets, transforms as T
from PIL import Image
from torchvision.utils import draw_bounding_boxes

alexnet = models.alexnet(pretrained=True)

app=Flask(__name__)

@app.route('/',methods=['GET'])

def hello():
    return render_template('home.html')

@app.route('/',methods=['POST'])

 
# def predict():
#     img_url=request.form.get('image')

#     response=requests.get(img_url)

#     transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
#     img_pil = Image.open(io.BytesIO(response.content))        #image goes here
#     img = transform(img_pil)

#     feed = torch.unsqueeze(img, 0)
#     out = alexnet(feed)

#     with open('./imagenet_classes.txt') as f:
#         classes = [line.strip() for line in f.readlines()]
    
#     _, indices = torch.sort(out, descending=True)
#     percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
#     cls = [(classes[idx], percentage[idx].item()) for idx in indices[0][:3]]

#     return render_template('home.html', pred=cls)


def predict():
    img_url=request.form.get('image')
    response=requests.get(img_url)
    
    coco_names = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))

    transform = T.Compose([
        T.ToTensor(),
    ])
    def predict(image, model, device, detection_threshold):
        image = transform(image).to(device)
        image = image.unsqueeze(0) 
        outputs = model(image) 
        pred_classes = [coco_names[i] for i in outputs[0]['labels'].cpu().numpy()]
        pred_scores = outputs[0]['scores'].detach().cpu().numpy()
        pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
        boxes = pred_bboxes[pred_scores >= detection_threshold].astype(np.int32)
        return boxes, pred_classes, outputs[0]['labels']

    def draw_boxes(boxes, classes, labels, image):
        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
        for i, box in enumerate(boxes):
            color = COLORS[labels[i]]
            cv2.rectangle(
                image,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                color, 2
            )
            cv2.putText(image, classes[i], (int(box[0]), int(box[1]-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, 
                        lineType=cv2.LINE_AA)
        return image

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    response = requests.get(img_url)
    image = Image.open(io.BytesIO(response.content))
    model.eval().to(device)
    boxes, classes, labels = predict(image, model, device, 0.8)
    image = draw_boxes(boxes, classes, labels, image)
    return render_template('home.html', pred=image)

if __name__ == '__main__':
    app.run(port=3000,debug=True)