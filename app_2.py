import numpy as np
import pandas as pd
import streamlit as st 
from sklearn import preprocessing
import pickle

import tensorflow as tf

# Load the model
model = tf.keras.models.load_model("c:\\Users\\dimsh\\my_app\\fine_tuned_model")


cols=['image_outcome','marital_status','occupation','health_record','race','gender','mass_size']    

def main(): 
    st.title("Breast Cancer Detection and Classification")
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Breast Cancer Detection and Classification</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    
    Name = st.text_input("Name","")
    age = st.text_input("Age","0") 
    image_outcome = st.selectbox(["Normal","Malignant","Benign"]) 
    marital_status = st.selectbox("marital_status",["Divorced","Married-AF-spouse","Married-civ-spouse","Married-spouse-absent","Never-married","Separated","Widowed"]) 
    occupation = st.selectbox("Occupation",["Adm-clerical","Armed-Forces","Craft-repair","Exec-managerial","Farming-fishing","Handlers-cleaners","Machine-op-ins","Other-service","Priv-house","Prof-specialty","Protective","Sales","Tech-support","Transport-moving"]) 
    health_record = st.selectbox("Health_record",["low_risk","high_risk","medium_risk"]) 
    race = st.selectbox("Race",["Amer Indian Eskimo","Asian Pac Islander","Black","Other","White"]) 
    gender = st.selectbox("Gender",["Female","Male"])
    mass_size = st.selectbox("mass_size",["minute","lump","small","big","medium","large"])

    
    if st.button("Predict"): 
        features = [['image_outcome','marital_status','occupation','health_record','race','gender','mass_size']]
        data = {'Name': str(Name),'age': int(age), 'image_outcome': image_outcome, 'marital_status': marital_status, 'occupation': occupation,'health_record': health_record,'gender': gender, 'mass_size': 'mass_size', 'race': race,}
        print(data)
        df=pd.DataFrame([list(data.values())], columns=['Name','age','image_outcome','marital-status','occupation','health_record','race','gender','mass_size'])
                
        category_col =['image_outcome','marital-status','occupation','health_record','race','gender','mass_size']
        for cat in encoder_dict:
            for col in df.columns:
                le = preprocessing.LabelEncoder()
                if cat == col:
                    le.classes_ = encoder_dict[cat]
                    for unique_item in df[col].unique():
                        if unique_item not in le.classes_:
                            df[col] = ['Unknown' if x == unique_item else x for x in df[col]]
                    df[col] = le.transform(df[col])
            
        features_list = df.values.tolist()      
        prediction = model.predict(features_list)
    
        output = int(prediction[0])
        if output == 1:
            text = "normal"
        else:
            text = "cancerous"

        st.success('The Patient image is {}'.format(text))
      
if __name__=='__main__': 
    main()