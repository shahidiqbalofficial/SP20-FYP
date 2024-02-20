# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 23:03:36 2023

@author: ATECH
"""

 
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json
import uvicorn
from pyngrok import ngrok
from fastapi.middleware.cors import CORSMiddleware
import nest_asyncio

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class model_input(BaseModel):

    age : int
    sex : int
    cp : int
    trestbps : int
    chol : int
    fbs : int
    restecg : int
    thalach : int
    exang : int
    oldpeak : float
    slope : int
    ca : int
    thal : int

# loading the saved model
diabetes_model = pickle.load(open('D:\SP20-FYP\Final Year Project\ML and AI\Models\multiple disease predict\saved models\heart_model.sav', 'rb'))

@app.post('/heart_prediction')
def diabetes_predd(input_parameters : model_input):

    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)

    age = input_dictionary['age']
    sex = input_dictionary['sex']
    cp = input_dictionary['cp']
    tbps = input_dictionary['trestbps']
    chol = input_dictionary['chol']
    fbs = input_dictionary['fbs']
    rcg = input_dictionary['restecg']
    tch = input_dictionary['thalach']
    exg = input_dictionary['exang']
    olp = input_dictionary['oldpeak']
    sp = input_dictionary['slope']
    ca = input_dictionary['ca']
    thal = input_dictionary['thal']


    input_list = [age, sex, cp, tbps, chol, fbs, rcg, tch, exg, olp, sp, ca, thal]

    prediction = diabetes_model.predict([input_list])

    if (prediction[0] == 0):
        return 'The person has not Heart Disease'
    else:
        return 'The person has Heart Disease'

ngrok_tunnel = ngrok.connect(8000)
print('Public URL:', ngrok_tunnel.public_url)
nest_asyncio.apply()
uvicorn.run(app, port=8000)