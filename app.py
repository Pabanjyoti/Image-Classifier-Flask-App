# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 13:17:51 2021

@author: DELL
"""
import alexnet
import fasterRcnn
from flask import Flask, render_template, request

app=Flask(__name__)

@app.route('/',methods=['GET'])

def hello():
    return render_template('home.html')

@app.route('/',methods=['POST'])


def predict():

    img_url = request.form.get('image')
    nnModel = request.form.get('nnModel')

    if nnModel == 'alexnet':
        cls = alexnet.alex(img_url)
    elif nnModel == 'fasterRcnn':
        # cls = fasterRcnn.faster(img_url)
        cls = alexnet.alex(img_url)

    #fasterRcnn.faster(img_url)                 #for faster rcnn

    return render_template('home.html', pred=cls)

if __name__ == '__main__':
    app.run(port=3000,debug=True)