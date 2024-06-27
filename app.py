import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
import yfinance as yf
st.title('Stock Price Prediction')
start = '2014-01-01'
end = '2024-01-01'

user_input = st.text_input('Enter Stock Ticker','AAPL')
df = yf.download(user_input, start=start, end=end)

st.subheader('Data from 2014 - 2024')
st.write(df.describe())

st.subheader('Closing Price Vs Time Chart')

fig = plt.figure(figsize = (12,6))
plt.plot(df.close)
st.pyplot(fig)

ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()

#Splitting Data
train = pd.DataFrame(df['Close'][0:int(len(df)*0.7)])
test = pd.DataFrame(df['Close'][int(len(df)*0.7):int(len(df))])

#Scale Down data for LSTM
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

train_arr = scaler.fit_transform(train)

x_train =[]
y_train =[]

for i in range(100, train_arr.shape[0]):
   x_train.append(train_arr[i-100:i])
   y_train.append(train_arr[i,0])
x_train = np.array(x_train)
y_train = np.array(y_train)

model = load_model("keras_model.h5")

past_100_days = train.tail(100)

final_df = pd.concat([past_100_days, test], ignore_index=True)

input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100,input_data.shape[0]):
  x_test.append(input_data[i-100:i])
  y_test.append(input_data[i,0])

x_test = np.array(x_test)
y_test = np.array(y_test)

y_predicted = model.predict(x_test)

factor = scaler.scale_
y_predicted = y_predicted/factor
y_test = y_test/factor

fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_predicted,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
