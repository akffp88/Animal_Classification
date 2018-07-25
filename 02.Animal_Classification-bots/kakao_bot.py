# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 14:49:02 2018

@author: LEE
"""

from flask import Flask
from flask import request
from flask import jsonify
from flask import json

import myProcessing
import pandas as pd
import numpy as nd

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "Hello World"


@app.route("/keyboard")
def keyboard():
        content = {
            'type' : 'buttons',
            'buttons' : ['Bear', 'Dolphin', 'Duck', 'Elephant', 'Frog', 'Gorilla', 'Honeybee', 'Lobster', 'Octopus', 'Seahorse', 'Seasnake']
            }
        return jsonify(content)

@app.route("/message",methods=['GET', 'POST'])
def message():
        print(request.data)
        data = json.loads(request.data)
        content = data["content"]

        if content == "Bear":        
                raw_data = "1,0,0,1,0,0,1,1,1,1,0,0,4,0,0,1"
        elif content == "Dolphin":
                raw_data = "0,0,0,1,0,1,1,1,1,1,0,1,0,1,0,1"
        elif content == "Duck":
                raw_data = "0,1,1,0,1,1,0,0,1,1,0,0,2,1,0,0"
        elif content == "Elephant":
                raw_data = "1,0,0,1,0,0,0,1,1,1,0,0,4,1,0,1"
        elif content == "Frog":
                raw_data = "0,0,1,0,0,1,1,1,1,1,1,0,4,0,0,0"
        elif content == "Gorilla":
                raw_data = "1,0,0,1,0,0,0,1,1,1,0,0,2,0,0,1"
        elif content == "Honeybee":
                raw_data = "1,0,1,0,1,0,0,0,0,1,1,0,6,0,1,0"
        elif content == "Lobster":
                raw_data = "0,0,1,0,0,1,1,0,0,0,0,0,6,0,0,0"
        elif content == "Octopus":
                raw_data = "0,0,1,0,0,1,1,0,0,0,0,0,8,0,0,1"
        elif content == "Seahorse":
                raw_data = "0,0,1,0,0,1,0,1,1,0,0,1,0,1,0,0"
        elif content == "Seasnake":
                raw_data = "0,0,0,0,0,1,1,1,1,0,1,0,0,1,0,0"
        else :
                raw_data = ""

        result = myProcessing._get_response_(content)
        
        result = pd.Series(result).to_json(orient='values')
        
        if result == '[1]':
                result = "Mammal"
        elif result == '[2]':
                result = "Bird"
        elif result == '[3]':
                result = "Reptile"
        elif result == '[4]':
                result = "Fish"
        elif result == '[5]':
                result = "Amphibian"
        elif result == '[6]': 
                result = "Bug"
        elif result == '[7]':
                result = "Invertebrate"
        else :
                result = result

        text = raw_data + "\n" +  result

        response ={
                "message" :{
                        "text" : text
                },

                "keyboard" : {
                        'type' : 'buttons',
                        'buttons' : ['Bear', 'Dolphin', 'Duck', 'Elephant', 'Frog', 'Gorilla', 'Honeybee', 'Lobster', 'Octopus', 'Seahorse', 'Seasnake'] 
                }
        }

        response = json.dumps(response)

        return response

if __name__ == "__main__":
    myProcessing._setup_()
    app.run(host="0.0.0.0", port=5000)