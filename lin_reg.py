import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
# %matplotlib inline

st.write("""
# Simple Student performance prediction App

This app predicts the **Percentage Score** of a student based on **Study Hours**
""")

st.header("Student_Score DataSet")
st.text("")

Data_url = "http://bit.ly/w-data"
df = pd.read_csv(Data_url)

st.dataframe(df)

x = df.iloc[:, :-1].values  
y = df.iloc[:, 1].values  

from sklearn.model_selection import train_test_split as tts
x_train, x_test, y_train, y_test = tts(x,y,test_size=0.2,random_state=0) 

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)

line = model.coef_*x+model.intercept_

st.header("Data Visalization")
st.text("")

fig=plt.figure()
ax=fig.add_subplot(111)
ax.scatter(df.Hours,df.Scores,color='red',marker='*')
ax.set_xlabel('Study Hours')
ax.set_ylabel('Percentage Scores')
ax.plot(x,line)

st.pyplot(fig)

st.header("Model Score")
st.text("")

st.write('Overall Performance based on training : ',model.score(x_test,y_test))

y_pred = model.predict(x_test)

from sklearn import metrics  
st.write('Mean Absolute Error : ',metrics.mean_absolute_error(y_test, y_pred)) 
st.write('Mean Squared Error : ',metrics.mean_squared_error(y_test, y_pred)) 
st.write('Root Mean Squared Error : ',np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

st.header("Insert Study Hours : ")
hours = st.number_input('')

result = st.button("Predict")

if result:
    hours = np.array(hours)
    hours = hours.reshape(-1,1)
    prediction = model.predict(hours)
    st.write('Predicting Student Score : ',prediction)


