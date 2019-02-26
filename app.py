import os
import pickle
import re

from flask import Flask, request, jsonify

brand = [1,2,4,5,3,6]
model = [22,0,25,27,39,32,3,275,10,28,11,26,14,21,24,13]
spending = ['40k','00','20k']
regdate = [2007,2017,2015,2018,2011,2014,2010,2013,2008,2009,1980,2002,1981,2012,2006,2001,2000,1999,2004,2016,1998,2003,1997,2005,1990,1993,1988,1994,1991,1995,1982,1996,1992,1984,1987,1989,1983,1986,1985,2019]
mileage = ['05k','60k','15k_30k','05k_15k','30k_60k']
region = [13,2,10,7,3,12,9,1,5,8,6,4,11]
mc_type = [2.0,1.0,3.0]
mc_capacity = [2.0,3.0,6.0,5.0,1.0,4.0]

def init():
    return ""

app = Flask(__name__, static_folder='static')

@app.route('/hello')
def index():
    return jsonify({
        'message': 'hello world'
    })


@app.route('/classify', methods=['POST'])
def classify():
    return jsonify({
        'message': 'hello world'
    })

app.run(debug=True,host='0.0.0.0',port=9000)
