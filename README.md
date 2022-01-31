# Image-Classifier-Flask-App

It is a tool to compare differnt pre-trained Machine Learning Models using Pytorch.
Written in Python using Flask.

## The API

### The API expects a json request as below:
```json
{
    "url": "https://www.tensorflow.org/tutorials/images/classification_files/output_N1loMlbYHeiJ_0.png",
    "model": "squeezenet"
}
```

where `url` can be a url for a image file and `model` can be anything from below array:
```py
["alexnet", "resnet", "squeezenet", "vgg", "densenet", "googlenet", "shufflenet", "mobilenet", "resnext", "wide_resnet", "mnasnet", "efficientnet", "regnet_x", "regnet_y"]
```
### The Response of the API request is:
```json
{
    "clsName1": "bee",
    "clsName2": "bubble",
    "clsName3": "ant, emmet, pismire",
    "img_url": "https://www.tensorflow.org/tutorials/images/classification_files/output_N1loMlbYHeiJ_0.png",
    "nnModel": "squeezenet",
    "percent1": 85.892822265625,
    "percent2": 2.063873529434204,
    "percent3": 1.8604017496109009
}
```
where `clsName` indicates the detected object and `percent` indicates respective confidance percentage.

### Here is a screenshot of the API request and it's response in POSTMAN:
![screenshot](https://i.imgur.com/XxzQZ33.png)

## Colab Notebook for demo:
[Click this link to open Colab Notebook. Run each cell one after another.](https://colab.research.google.com/drive/13-uIcgEsFGA3ZNgqAPxQgQ82I5Uk9lpS?usp=sharing)