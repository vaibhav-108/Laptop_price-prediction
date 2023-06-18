from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import pickle

with open('model.pkl', 'rb') as file:
    mod = pickle.load(file)

# model = pickle.load(open('df.pkl', 'rb'))
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/Laptop', methods=['POST'])
def laptop():
    data = request.form

    company = str(data.get('Company'))
    Typename = str(data.get('TypeName'))
    Ram = int(data.get('ram'))
    Weight = float(data.get('weight'))
    Touchscreen = str(data.get('touchscreen'))
    IPS = str(data.get('Ips'))
    CPU_processor = float(data.get('Cpu_Processor'))
    CPU_Brand = str(data.get('Cpu_Brand'))
    ssd = int(data.get('SSD'))
    hdd = int(data.get('HDD'))
    Flash_memory = int(data.get('Flash_Storage'))
    hybrid = str(data.get('Hybrid'))
    os = str(data.get('OS'))
    GPU = str(data.get('GPU_Model'))
    size = float(data.get('Screen_Size'))
    resolution = str(data.get('Resolution'))

    Touchscreen = 1 if Touchscreen == 'yes' else 0
    IPS = 1 if IPS == 'yes' else 0
    hybrid = 1 if hybrid == 'yes' else 0

    x_resolution = resolution.split('x')[0]
    y_resolution = resolution.split('x')[1]

    PPI = np.sqrt(np.square(int(x_resolution)) +
                  np.square(int(y_resolution))/size)

    query = np.array([company, Typename, Ram, Weight, Touchscreen, IPS, CPU_processor, CPU_Brand,
                      ssd, hdd, Flash_memory, hybrid, GPU, os, PPI])

    query = query.reshape(1, 15).astype('object')
    pred = int(np.exp(mod.predict(query)[0]).tolist())
    # print(pred)
    # return jsonify(pred)

    return render_template('index.html', prediction=pred)


if __name__ == '__main__':
    app.run(debug=True)
