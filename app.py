# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 13:17:51 2021
@author: DELL
"""
import cnn
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

    if nnModel == 'fasterRcnn':
        cls = fasterRcnn.faster(img_url)
        return render_template('home.html', img_url= img_url, rCnnPred=cls)
    else:
        cls = cnn.detect(img_url)
        return render_template('home.html', img_url= img_url, clsName1=cls[0][0], percent1=cls[0][1], clsName2=cls[1][0], percent2=cls[1][1], clsName3=cls[2][0], percent3=cls[2][1])
                 
if __name__ == '__main__':
    app.run(port=3000,debug=True)