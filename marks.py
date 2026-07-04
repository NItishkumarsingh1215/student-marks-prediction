import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
st.set_page_config(page_title="student marks prediction",layout="centered")
st.title("student marks prediction")
st.write("this app predicts student marks based on the number of hours studied")

df = pd.read_csv("marks.csv ")
x = df[["Study_hours"]]
y = df[["Marks"]]
x_train,x_test,y_train,y_test = train_test_split(x , y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train,y_train)
pred = model.predict(x_test)

score = r2_score(y_test,pred)
with st.container(border=True):
  st.subheader("student marks prediction")
  hours = st.number_input("enter the number of hours studied:",min_value=0.0,max_value=24.0,step=0.5)
  if st.button("predict marks"):
    prediction = model.predict([[hours]])
    st.success(f"predicted marks:{prediction.item():.2f}")
    st.info(f"model accuracy score:{score:.4f}")

    fig, ax = plt.subplots (figsize=(6,3))
    ax.scatter(x,y, color='blue',label="student data",s=80)
    ax.plot(x, model.predict(x),color='red',label='Regression Line',linewidth=1)
    ax.set_title("study hours vs marks")
    ax.set_ylabel("marks")
    ax.legend()

    st.pyplot(fig)