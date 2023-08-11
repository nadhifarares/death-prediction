import streamlit as st
import pandas as pd
import numpy  as np
import joblib

st.header('Death Prediction at Rares Hospital')
st.write("""
Memprediksikan apakah seorang pasien akan meninggal atau tidak berdasarkan riwayat penyakitnya.
""")

@st.cache_data
def fetch_data():
    df = pd.read_csv('h8dsft_P1G3_nadhifasafira.csv')
    return df

df = fetch_data()

age = st.number_input('age', value=0)
sex = st.selectbox('sex', df['sex'].unique())
time = st.number_input('time', value=0)
smoking = st.selectbox('smoking', df['smoking'].unique())
anaemia = st.selectbox('anaemia', df['anaemia'].unique())
creatinine_phosphokinase = st.number_input('creatinine_phosphokinase', value=0)
diabetes = st.selectbox('diabetes', df['diabetes'].unique())
ejection_fraction = st.number_input('ejection_fraction', value=0)
high_blood_pressure = st.selectbox('high_blood_pressure', df['high_blood_pressure'].unique())
platelets = st.number_input('platelets', value=0)
serum_creatinine = st.number_input('serum_creatinine', value=0)
serum_sodium = st.number_input('serum_sodium', value=0)



data = {
    'age': age,
    'sex' : sex,
    'time': time,
    'smoking': smoking,
    'anaemia': anaemia,
    'creatinine_phosphokinase': creatinine_phosphokinase,
    'diabetes': diabetes,
    'ejection_fraction': ejection_fraction,
    'high_blood_pressure': high_blood_pressure,
    'platelets': platelets,
    'serum_creatinine': serum_creatinine,
    'serum_sodium': serum_sodium
}
input = pd.DataFrame(data, index=[0])

st.subheader('Patient`s record')
st.write(input)

load_model = joblib.load("death_pred.pkl")

if st.button('Predict'):
    prediction = load_model.predict(input)

    if prediction == 1:
        prediction = 'Dead'
    else:
        prediction = 'Alive'

    st.write('Based on patient input, the placement model predicted: ')
    st.write(prediction)