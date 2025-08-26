import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LeakyReLU

model = load_model("model.keras", custom_objects={"LeakyReLU": LeakyReLU})
with open("LabelEncoderGender.pkl","rb") as f:
    label_encoder_gender = pickle.load(f)

with open("OneHotEncodergeography.pkl","rb") as f:
    one_hot_encoder_geo = pickle.load(f)

with open("StandardScaler.pkl","rb") as f:
    standard_scaler = pickle.load(f)

st.title("Customer Churn Prediction")
st.caption("Enter Customer details to see whether the customer will churn or not")



creditscore = st.slider("Credit Score",0,1000,250)
Geography = st.selectbox("Geography",one_hot_encoder_geo.categories_[0])
Gender = st.selectbox('Gender',label_encoder_gender.classes_)
Age = st.slider("Age",1,100,22)
Tenure = st.number_input("Tenure",0,10,1)
Balance = st.number_input("Balance",0.,500000000.0,0.0)
Numofprodcut = st.number_input("NumberOfProducts",1,4,1)
hasactivecard = st.selectbox("HasActiveCard" ,[0,1])
isactivemember = st.selectbox("IsActiveMember",[0,1])
estimatesalary = st.number_input("EstimatedSalary",0.,500000000.0 , 0.)

data =      pd.DataFrame({"CreditScore" : [creditscore],
            "Gender" : label_encoder_gender.transform([Gender]),
            "Age" : [Age],
            "Tenure" : [Tenure],
            "Balance" : [Balance],
            "NumOfProducts" : [Numofprodcut],
            "HasCrCard" : [hasactivecard],
            "IsActiveMember" : [isactivemember],
            "EstimatedSalary" :  [estimatesalary]})

geo = one_hot_encoder_geo.transform([[Geography]]).toarray()
geo = pd.DataFrame(geo,columns = one_hot_encoder_geo.get_feature_names_out(["Geography"]))
geo = geo.astype(float)
df = pd.concat([data,geo],axis = 1)
df = df[standard_scaler.feature_names_in_] 
df_scaled = standard_scaler.transform(df)







if st.button("Predict"):
    pre = model.predict(df_scaled)
    pre_pro = pre[0][0]
    st.write(f"Churn Probability : {pre_pro:.4f}")
    if pre_pro > 0.5:
        st.success("Customer will Churn")
    else:
        st.info("Customer will not Churn")