import streamlit as st
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.title("Laptop Price Predict")

# Company
company = st.selectbox('company', df['Company'].unique())

# Brand_Type
TypeName = st.selectbox('Brand_Type', df['TypeName'].unique())

# Ram
Ram = st.selectbox("Ram", df['Ram'].value_counts().index.sort_values())
# Ram = st.selectbox("Ram", [0, 2, 4, 8, 16, 32])

# Weight
weight = st.number_input('Weight')

# Touchscreenl
Touchscreen = st.selectbox("Touchscreen", ['Yes', 'No'])

# IPS
IPS = st.selectbox("IPS", ['Yes', 'No'])

# CPU_processor
Cpu_Processor = st.selectbox(
    "cpu_processor", df['Cpu_Processor'].value_counts().index.sort_values())
# Cpu_Processor = st.selectbox("cpu_processor", [1.8, 2.3, 2.5, 2.7])

# cpu_brand
Cpu_Brand = st.selectbox('cpu_brand', df['Cpu_Brand'].unique())

# SSD
SSD = st.selectbox("SSD", df['SSD'].value_counts().index.sort_values())
# SSD = st.selectbox("SSD", [0, 2, 8, 16, 32, 64])

# HDD
HDD = st.selectbox("HDD", df['HDD'].value_counts().index.sort_values())
# HDD = st.selectbox("HDD", [0, 2, 8, 16, 32, 64])

# Flash_Storage
Flash_storage = st.selectbox(
    'Flash_storage', df['Flash_storage'].value_counts().index.sort_values())
# Flash_storage = st.selectbox('Flash_storage', [0, 32, 64, 256])

# Hybrid
Hybrid = st.selectbox("Hybrid", ['Yes', 'No'])

# GPU model
Gpu_model = st.selectbox('GPU Model', df['Gpu_model'].unique())

# OS
OS = st.selectbox('OS', df['OS'].unique())

# screen Size
size = st.number_input("Screen Size")

resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', '32001800',
                                                '2880x1800', '2560x1600', '2560x1440', '230x1440'])

x_resolution = resolution.split('x')[0]
y_resolution = resolution.split('x')[1]

# PPI
PPI = np.sqrt(np.square(int(x_resolution))+np.square(int(y_resolution))/size)


if st.button('Price Predict'):
    PPI = None

    x_resolution = resolution.split('x')[0]
    y_resolution = resolution.split('x')[1]

# PPI
    PPI = np.sqrt(np.square(int(x_resolution)) +
                  np.square(int(y_resolution))/size)
    # Touchscreen = 1 if Touchscreen == 'Yes' else 0
    #   or
    Touchscreen = [1 if Touchscreen == 'Yes' else 0]
    IPS = [1 if IPS == 'Yes' else 0]
    Hybrid = [1 if Hybrid == 'Yes' else 0]

    querry = np.array([company, TypeName, Ram, weight, Touchscreen[0], IPS[0],
                       Cpu_Processor, Cpu_Brand, SSD, HDD, Flash_storage, Hybrid[0], Gpu_model, OS, PPI])

    querry = querry.reshape(1, 15)
    querry = querry.astype('object')
    # st.title(querry)  ----> for checking purpose

st.title('Predicted Price {company} of ' + company + TypeName +
         str(int(np.exp(model.predict(querry)[0]))))
