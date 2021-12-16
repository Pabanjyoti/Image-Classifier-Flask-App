# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 13:17:51 2021

@author: DELL
"""

from flask import Flask, render_template, request


app=Flask(__name__)

@app.route('/',methods=['GET'])

def hello():
    return render_template('home.html')

@app.route('/',methods=['POST'])

def predict():
    image=request.files['image']
    image_path="./images/"+image.filename
    image.save(image_path)
    return render_template('home.html')

if __name__ == '__main__':
    app.run(port=3000,debug=True)