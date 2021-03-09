from flask import Flask
import pandas as pd
import numpy as np
import yfinance as yf
from keras.models import load_model
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler,StandardScaler


app = Flask(__name__)
@app.route("/")
def main():
  model=load_model('my_model.h5') 
  modelhigh=load_model('my_modelhigh.h5') 
  modellow=load_model('my_modellow.h5') 
  nifty = yf.Ticker("^NSEI")
  usdinr = yf.Ticker("INR=X")
  niftyy = nifty.history(period="3mo")
  usdinrr = usdinr.history(period="3mo")
  dataframe = pd.DataFrame()




  dataframe['niftyyClose']= (niftyy.Close)
  dataframe['niftyyOpen'] = (niftyy.Open)  
  dataframe['niftyyHigh'] = (niftyy.High)
  dataframe['niftyyLow'] = (niftyy.Low)
  dataframe['niftyyVolume'] = (niftyy.Volume)
  dataframe['usdClose']= (usdinrr.Close)
  dataframe['usdOpen'] = (usdinrr.Open)  
  dataframe['usdHigh'] = (usdinrr.High)
  dataframe['usdLow'] = (usdinrr.Low)


  df = dataframe
  df = df.iloc[-61:]
  df.iloc[-1] = np.NaN
  scaler = StandardScaler()
  scaledinput = scaler.fit_transform(df)
  scaledoutput = scaledinput[:,0]
  test_generator = TimeseriesGenerator(scaledinput,scaledoutput,length=60,sampling_rate=1,shuffle= False)

  predictions = model.predict(test_generator)
  df_pred = pd.concat([pd.DataFrame(predictions),pd.DataFrame(scaledinput[:,1:][60:])],axis = 1)
  predictioninv = scaler.inverse_transform(df_pred)
  df_final = df[predictions.shape[0]*-1:]
  df_final['clopenpred'] = predictioninv[:,0]
  df_final[['clopenpred','niftyyClose']]
  result = df_final['clopenpred'].iloc[-1]

  
  dataframe = dataframe[['niftyyLow','niftyyClose','niftyyOpen','niftyyHigh','niftyyVolume','usdClose','usdOpen','usdHigh','usdLow']]


  df = dataframe

  df = df.iloc[-61:]
  df.iloc[-1] = np.NaN
  scaler = StandardScaler()
  scaledinput = scaler.fit_transform(df)
  scaledoutput = scaledinput[:,0]
  test_generator = TimeseriesGenerator(scaledinput,scaledoutput,length=60,sampling_rate=1,shuffle= False)

  predictions = modellow.predict(test_generator)
  df_pred = pd.concat([pd.DataFrame(predictions),pd.DataFrame(scaledinput[:,1:][60:])],axis = 1)
  predictioninv = scaler.inverse_transform(df_pred)
  df_final = df[predictions.shape[0]*-1:]
  df_final['lowpred'] = predictioninv[:,0]
  df_final[['lowpred','niftyyLow']]
  resultlow = df_final['lowpred'].iloc[-1]


  dataframe = dataframe[['niftyyHigh','niftyyClose','niftyyOpen','niftyyLow','niftyyVolume','usdClose','usdOpen','usdHigh','usdLow']]

  df = dataframe
  df = df.iloc[-61:]
  df.iloc[-1] = np.NaN
  scaler = StandardScaler()
  scaledinput = scaler.fit_transform(df)
  scaledoutput = scaledinput[:,0]
  test_generator = TimeseriesGenerator(scaledinput,scaledoutput,length=60,sampling_rate=1,shuffle= False)

  predictions = modelhigh.predict(test_generator)
  df_pred = pd.concat([pd.DataFrame(predictions),pd.DataFrame(scaledinput[:,1:][60:])],axis = 1)
  predictioninv = scaler.inverse_transform(df_pred)
  df_final = df[predictions.shape[0]*-1:]
  df_final['highpred'] = predictioninv[:,0]
  df_final[['highpred','niftyyHigh']]
  resulthigh = df_final['highpred'].iloc[-1]
  return f"close could be {result} high could be {resulthigh} low could be {resultlow} "

if __name__ == "__main__":
  app.run()
