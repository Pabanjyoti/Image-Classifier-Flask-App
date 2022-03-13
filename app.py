# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 13:17:51 2021

@author: DELL
"""
import cnn
import urlValid

from flask import Flask, render_template, request, jsonify

app=Flask(__name__)

@app.route('/',methods=['GET'])
def hello():
    return render_template('index.html')

@app.route('/api/pred/', methods=['GET', 'POST'])
def pred():

    img_url = request.json['url']
    nnModel = request.json['model']

    models = ["alexnet", "resnet", "squeezenet", "vgg", "densenet", "googlenet", "shufflenet", "mobilenet", "resnext", "wide_resnet", "mnasnet", "efficientnet", "regnet_x", "regnet_y"]

    if nnModel not in models:
        return jsonify(nnModel=nnModel, img_url=img_url,
                   response = 'not a valid model')        

    url_check = urlValid.valid_url(img_url)

    if url_check != 'invalid_image_url':
        predResult= cnn.detect(img_url, nnModel)
        return jsonify(nnModel=nnModel, img_url=img_url, response = url_check,
                   clsName1=predResult[0][0], percent1=predResult[0][1],
                   clsName2=predResult[1][0], percent2=predResult[1][1],
                   clsName3=predResult[2][0], percent3=predResult[2][1])
    else:
        return jsonify(nnModel=nnModel, img_url=img_url,
                   response = 'not a valid image url')

@app.route('/',methods=['POST'])
def predict():
    return ('', 204)
        
if __name__ == '__main__':
    app.run(port=3000,debug=True)