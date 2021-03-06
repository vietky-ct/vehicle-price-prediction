import os
import pickle
import re

from flask import Flask, request, jsonify
from sklearn.preprocessing import OneHotEncoder
import math
import numpy as np
import pandas as pd

percentageOffset = 20
priceReducer = 1000000

brand = np.asarray([3,6,1,2,5,4])
model = np.asarray([0,275,11,3,14,22,80,25,10,13,27,78,19,1,24,39,28,26,32,21,293])
spending = np.asarray(['0'])
regdate = np.asarray([2010,2009,2008,1999,2013,2012,2016,2007,2014,1997,2001,2003,1985,1998,2017,2011,2005,1995,2006,2015,2000,2002,2004,1996,1981,1986,1980,1989,1994,2018,1990,1987,1992,1993,1991,1983,1982,1984,1988,2019])
mileage = np.asarray(['None','15k_30k','30k_60k','60k','05k','05k_15k'])
region = np.asarray([12,2,13,4,3,1,5,6,8,9,7,10,11])
mc_type = np.asarray([1.0,2.0,3.0])
mc_capacity = np.asarray([3.0,2.0,1.0,5.0,6.0,4.0])

brandMapping = {'honda': 1, 'yamaha': 2, 'piaggio': 3, 'kawasaki': 17, 'sym': 5, 'suzuki': 4, 'hãng khác': 34, 'bazan': 7, 'ducati': 12, 'rebelusa': 26, 'halim': 14, 'daelim': 11, 'kymco': 21, 'brixton': 35, 'bmw': 9, 'benelli': 8, 'hyosung': 16, 'keeway': 18, 'visitor': 33, 'victory': 32, 'moto guzzi': 36, 'harley davidson': 15, 'euro reibel': 13, 'ktm': 20, 'mv agusta': 24, 'aprilia': 6, 'sachs': 28, 'sanda': 29, 'triumph': 30, 'lambretta': 22, 'kengo': 19, 'vento': 31, 'cr&s': 10, 'peugeot': 37, 'regal raptor': 27, 'norton': 25, 'malaguti': 23}
modelMapping = {'67': 1, 'future': 13, 'sirius': 27, 'lx': 79, 'wave': 22, 'sh': 19, 'jupiter': 28, 'cub': 10, 'max': 97, 'vespa': 80, 'yass': 36, 'shark': 51, 'dream': 11, 'air blade': 3, 'attila': 39, 'nouvo': 26, 'ps': 16, 'liberty': 78, 'dòng khác': 275, 'luvias': 31, 'vision': 21, 'exciter': 25, 'nozza': 33, 'click': 9, 'kawasaki': 94, 'nova': 336, 'pcx': 15, 'enjoy': 44, 'lead': 14, 'citi': 8, 'revo': 340, 'viva': 59, 'z1000': 104, 'scr': 18, 'msx 125': 292, 'acruzo': 351, 'grande': 280, 'sh mode': 279, 'mio': 32, 'dylan': 12, 'scoopy': 335, 'nvx': 370, 'ez': 45, 'winner': 293, 'hayate': 57, 'smash': 74, 'cb': 4, 'raider': 70, 'sport / xipo': 56, 'win': 23, 'yamaha r': 281, 'magic': 49, '@': 2, 'janus': 294, 'primavera': 84, 'cuxi': 29, 'spacy': 20, 'elizabeth': 347, 'angela': 40, 'rebell': 206, 'zip': 81, 'cbr': 5, 'candy hi': 127, 'z250': 339, 'axelo': 61, 'fz': 30, 'en': 65, 'fx125': 58, 'bx 150': 301, 'gz': 68, 'stinger': 75, 'impulse': 283, 'r nine t': 182, 'galaxy': 47, 'tnt': 312, 'gsx': 295, 'gd': 296, 'gn': 67, 'candy s': 128, 'fly': 82, 'venus': 286, 'blade': 333, 'monster': 161, 'taurus': 35, 'fiddle': 46, 'elegant': 43, 'rebel': 17, 'z800': 107, 'phoenix': 269, 'estrella': 93, 'best': 63, 'vegas': 357, 'joyride': 48, 'epicuro': 66, 'yaz': 37, 'tfx': 353, 'v7': 303, 'chaly': 7, 'cd': 6, 'star': 52, 'cello': 42, 'avitor': 334, 'bonus': 41, 'satria': 60, 'sanda boss': 50, 'sportster': 327, 'vs125': 321, '125/250': 187, 'husky': 69, 'notus': 205, 'z300': 105, 'sprint': 83, 'blackster': 191, 'z900': 297, 'xbike': 76, 'deluxe': 204, 'z125': 338, 'sapphire': 73, 'gts': 85, 'brutale 3 cylind': 258, 'scrambler': 299, 'jockey fi': 129, 'bn 600i': 180, 'mt': 355, 'bella': 62, 'k-pipe': 130, 'goebel': 345, 'bios': 356, 'c600': 318, 'gt': 87, 'duke 200': 163, 'r': 34, 'bs': 322, 'nm-x': 354, 'vespa s125': 86, 'gunner': 265, 'bx 125': 300, 'ninja': 98, 't15': 315, 'ns125': 199, 'like mmc': 132, 'z650': 298, 'royal': 72, 'like fi': 131, '125 classic': 267, 'amici': 343, 'symphony': 53, 'hypermotard': 288, 'wolf': 54, 'rgv': 71, 'solona': 337, 'crystal': 64, 'r250': 141, 'a-4': 193, 'medley': 89, 'w650': 102, 'bee 50 and bee 125': 216, 'daystar125fi': 198, 'e-bikes': 146, 's1000rr': 185, 'bn 302': 179, 'c650': 319, '899 panigale': 160, 'high-ball': 358, 'elite': 346, 'rc 200': 172, 'people 16 fi': 137, '102': 307, 'cdr': 202, 'zx6j': 109, 'street': 326, 'duke 390': 170, 'sr 125': 231, 'px': 90, 'many': 133, 'st': 325, 'vulcan': 100, 'besbi125': 197, 'amigo': 348, 'triumph bonneville t100': 242, 'b-bone': 196, 'et8': 88, 'streetfighter': 291, '302r': 313, 'versys': 99, 'xl 1200x forty-eight': 176, 'rc 390': 174, '390': 167, 'leo': 96, 'cuu': 156, 'gladiator xls': 249, 'a': 149, 'ksr': 95, 'e-five': 194, 'duke 250': 169, 'madass': 220, 'diavel': 287, '104': 309, 'helios': 268, 'zx10r': 108, '103': 308, 'r1200': 317, 's1000r': 184, 'z400': 106, 'f800': 320, 'magnum': 361, 's4': 195, 'rc 250': 173, 'dorsoduro 750 abs': 116, 'dragster': 260}

onehot_brand = OneHotEncoder().fit(brand.reshape(-1,1))
onehot_model = OneHotEncoder().fit(model.reshape(-1,1))
onehot_spending = OneHotEncoder().fit(spending.reshape(-1,1))
onehot_regdate = OneHotEncoder().fit(regdate.reshape(-1,1))
onehot_mileage = OneHotEncoder().fit(mileage.reshape(-1,1))
onehot_region =  OneHotEncoder().fit(region.reshape(-1,1))
onehot_mctype =  OneHotEncoder().fit(mc_type.reshape(-1,1))
onehot_mccapa =  OneHotEncoder().fit(mc_capacity.reshape(-1,1))

lgmod = pickle.load(open('./data/finalized_model.sav', 'rb'))

app = Flask(__name__, static_folder='static')

def cate_brand(x):
    x = brandMapping[x] if x in brandMapping else 0
    if x > 6:
        return 6
    return(x)


def cate_model(x):
    x = modelMapping[x] if x in modelMapping else 0
    if x not in model:
        return 0
    return(x)


#categorize duration
def cate_duration(x):
    x = int(x)
    if x >= 0 and x <= 48:
        value = '00_02_ngay'
    else:
        value = 'hon_2_ngay'
    return(value)

#categorize spending
def cate_spending(x):
    x = int(x)
    if x == 0:
        value = '00'
    elif x <= 20000:
        value = '20k'
    else:
        value = '40k'
    return(value)
#categorize mileage
def cate_mileage(x):
    x = int(x)
    if x >= 0 and x <= 5000 :
        value = '05k'
    elif x > 5000 and x <= 15000:
        value = '05k_15k'
    elif x > 15000 and x <= 30000:
        value = '15k_30k'
    elif x > 30000 and x <= 60000:
        value = '30k_60k'
    elif x > 60000:
        value = '60k'
    else:
        value = 'None'
    return value



def get_x_test_full(X_test):
    X_test_brand = onehot_brand.transform(X_test.brand.values.reshape(-1,1)).toarray()
    X_test_model = onehot_model.transform(X_test.model.values.reshape(-1,1)).toarray()
    # X_test_spending = onehot_spending.transform(X_test.spending.values.reshape(-1,1)).toarray()
    X_test_regdate = onehot_regdate.transform(X_test.regdate.values.reshape(-1,1)).toarray()
    X_test_mileage = onehot_mileage.transform(X_test.mileage.values.reshape(-1,1)).toarray()
    X_test_region = onehot_region.transform(X_test.region.values.reshape(-1,1)).toarray()
    X_test_mctype = onehot_mctype.transform(X_test.mc_type.values.reshape(-1,1)).toarray()
    X_test_mccapa = onehot_mccapa.transform(X_test.mc_capacity.values.reshape(-1,1)).toarray()
    X_test_price = X_test.price.values.reshape(-1, 1)
    X_test_full = np.concatenate([X_test_price,X_test_mileage,X_test_brand, X_test_model, X_test_regdate, X_test_region, X_test_mctype, X_test_mccapa], axis = 1)

    return X_test_full

@app.route('/hello')
def index():
    return jsonify({
        'message': 'hello world'
    })


@app.route('/classify', methods=['POST'])
def classify():
    arr = []
    for percent in range(-percentageOffset, percentageOffset, 1):
        price = (int(request.form['price']) * (100-percent)/100) / priceReducer
        arr.append({
            'brand_name': request.form['brand'].lower(),
            'model_name': request.form['model'].lower(),
            'spending': cate_spending(request.form['spending']),
            'regdate': request.form['regdate'],
            'mileage': cate_mileage(request.form['mileage']),
            'region': request.form['region'],
            'mc_type': request.form['mc_type'],
            'mc_capacity': request.form['mc_capacity'],
            'price': price
        })

    data = pd.DataFrame(arr)

    data['brand'] = data['brand_name'].apply(cate_brand)
    data['model'] = data['model_name'].apply(cate_model)
    # print(data.price.values)
    print(",".join([str(x) for x in data.price.values.tolist()]))

    data['price'] = np.log10(data['price'])
    data = data.drop(['model_name', 'brand_name'], axis=1)
    print(data)

    X_test_full = get_x_test_full(data)

    predictResult = lgmod.predict(X_test_full)
    predictResult_prob = lgmod.predict_proba(X_test_full)[:,0]

    maxIndex = np.argmax(predictResult_prob)
    minIndex = np.argmin(predictResult_prob)

    # valueIndex = np.argwhere(data['price'] == math.log10(int(request.form['price'])))

    print(",".join([str(x) for x in predictResult_prob.tolist()]))
    # predictResult_prob = lgmod.predict_proba(X_test_full)[:,1]

    print('predicted', predictResult_prob[maxIndex], predictResult_prob[minIndex])
    print('type ne', type(predictResult_prob), type(data.price), type(data.price.iloc[minIndex]), data.price.iloc[minIndex])
    # print('valueInedx', valueIndex)

    nDays = 'trong' if predictResult[maxIndex] == 0 else 'hơn'
    print('data.price.iloc[maxIndex]', math.pow(10, data.price.iloc[maxIndex]))
    return jsonify({
        'message': 'Bạn có {}% cơ hội để bán đc sản phẩm với giá {:,} {} 2 ngày'.format('%.2f' % (predictResult_prob[maxIndex]*100), int(math.pow(10, data.price.iloc[maxIndex])*priceReducer),nDays),
        # 'message2': 'Bạn có {}% cơ hội để bán đc sản phẩm với giá {:,} {} 2 ngày'.format('%.2f' % (predictResult_prob[valueIndex]*100), int(math.pow(10, data.price.iloc[valueIndex])*priceReducer),nDays),
        'price': data['price'].apply(lambda x: math.pow(10, x) * priceReducer).tolist(),
        'probs': predictResult_prob.tolist()
    })

app.run(debug=True,host='0.0.0.0',port=7777)
