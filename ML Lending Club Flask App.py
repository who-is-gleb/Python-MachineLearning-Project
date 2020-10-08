# A very simple Flask app to deplot Lending Club ML project to online.

from flask import Flask
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

from flask import Flask, jsonify, request
from flask_restful import Api, Resource, reqparse

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello from Flask!'

@app.route('/final',methods=['GET'])
def new_fund():
    #Get arguments
    parser=reqparse.RequestParser()
    parser.add_argument("s1", action="store",type=int)
    parser.add_argument("s2", action="store",type=int)
    parser.add_argument("s3", action="store",type=int)
    parser.add_argument("s4", action="store",type=int)
    parser.add_argument("s5", action="store",type=int)
    parser.add_argument("s6", action="store",type=int)
    parser.add_argument("s7", action="store",type=int)
    parser.add_argument("s8", action="store",type=int)
    parser.add_argument("s9", action="store",type=int)
    parser.add_argument("s10", action="store",type=int)

    args = parser.parse_args()

    # Loading data from main model
    X_train = pd.read_csv("/home/AJHoeft/mysite/X_train_select_2.csv")

    # Feature scaling
    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train))

    # Load the ML model
    filename = '/home/AJHoeft/mysite/finalized_model.model'
    loaded_model = pickle.load(open(filename,'rb'))

    df_new = [[args["s1"],args["s2"],args["s3"],args["s4"],args["s5"],args["s6"],args["s7"],args["s8"],args["s9"],args["s10"]]]
    X_test_scaled_new = pd.DataFrame(scaler.transform(df_new))
    predicted = loaded_model.predict(X_test_scaled_new)

    y_pred_proba_rf = loaded_model.predict_proba(X_test_scaled_new)

    y_pred = y_pred_proba_rf[0]

    if (y_pred[0] > .9):
        return('This individual is already likely to purchase a policy and does not need a letter')
    elif ((y_pred[0] >= .5) & (y_pred[0] <= .9)):
        return jsonify('This individual should receive a mailing')
    else:
        return jsonify('This individual should not receive a mailing')

__________________
