# %%writefile app.py%
import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


# loading the trained model
model = pickle.load(open('PickleModel.pkl','rb'))


def main():
    # front end elements of the web page
    html_temp = """ 
    <div style ="background-color:#002E6D;padding:20px;font-weight:15px"> 
    <h1 style ="color:white;text-align:center;"> Sport Prediction</h1> 
    </div> 
    """

    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html=True)
    default_value_goes_here = ""
    age = st.number_input('Please enter your age here: ')
    sex = st.selectbox(
     'Please select your sex here: ',
     ('Male', 'Female'))
    if sex == 'male':
        sex = 1
    else:
        sex = 0
        
    cp = st.selectbox(
     'Select your chest pain type here: ',
     ('Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'))
    if cp == 'Typical Angina':
        cp = 0
    elif cp == "Atypical Angina":
        cp = 1
    elif cp == "Non-anginal Pain":
        cp = 2
    else:
        cp = 3
        
    trestbps = st.number_input('Please enter your resting blood pressure (in mm Hg on admission to the hospital): ')
    chol = st.number_input('Please enter serum cholestoral in mg/dl: ')
    fbs = st.selectbox(
     'Please select your fasting blood sugar > 120 mg/dl: ',
     ('True', 'False'))
    if fbs == "True":
        fbs = 1
    else:
        fbs = 0
    restecg = st.selectbox(
     'Please select your resting electrocardiographic results: ',
     ('Normal', 'Having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)', 'Showing probable or definite left ventricular hypertrophy by Estes criteria'))
    if restecg == "Normal":
        restecg = 0
    elif restecg == "Having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)":
        restecg = 1
    else:
        restecg = 2
      
    thalach = st.number_input('Please enter your maximum heart rate achieved: ')
    exang = st.selectbox(
     'Please select your exercise induced angina: ',
     ('Yes', 'No'))
    if exang == "Yes":
        exang = 1
    else:
        exang = 0
    oldpeak = st.number_input('Please enter ST depression induced by exercise relative to rest: ')
    slope = st.selectbox(
     'Please select the slope of the peak exercise ST segment: ',
     ('Upsloping', 'Flat', 'Downsloping'))
    if slope == "Upsloping":
       slope = 0
    elif slope == "Flat":
       slope = 1
    else:
       slope = 2
    ca = st.number_input('Please enter your number of major vessels (0-3) colored by flourosopy: ')
    thal = st.selectbox(
     'Please select the slope of the peak exercise ST segment: ',
     ('Normal', 'Fixed defect', 'Reversable defect'))
    if thal == "Normal":
       thal = 0
    elif thal == "Fixed defect":
       thal = 1
    else:
       thal = 2
    attributes = [age, sex, cp, trestbps, chol, fbs, restecg, thalach,
       exang, oldpeak, slope, ca, thal]
     
    attributes = np.array(attributes)
    attributes = np.array([attributes])
#     mean = attributes.mean(axis=0)
#     attributes -= mean
#     std = attributes.std(axis=0)
#     attributes /= std
#     st.write(attributes)
    
    result = ""
#     #
    if st.button("Predict"):
#       arr = dataframe.columns

#       for i in arr:
#           notnull = dataframe[i][dataframe[i].notnull()]
#           min = notnull.min()
#           dataframe[i].replace(np.nan, min, inplace=True)

#       scaler = StandardScaler()
#       scaler.fit(dataframe)
#       featureshost = scaler.transform(dataframe)
      prediction = np.round(model.predict(attributes)).astype(int)

      result = prediction
      st.write(result)


if __name__ == '__main__':
    main()
