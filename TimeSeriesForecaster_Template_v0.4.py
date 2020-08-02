# -*- coding: utf-8 -*-
#########################
#Time Series Prediction Template v0.4
#Ali Nagaria
#Spring 2020
########################


import pandas as pd
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


TrainFile='TrainSet'
TestFile='TestSet'


#HELPER FUNCTIONS
def MakeCO2Plot(df=[],xlabel='Time',ylabel='CO2 (PPM)',title=''):
    x=df.datetime
    y=df.co2
    fig, ax = plt.subplots()
    ax.plot(x, y)
    fig.autofmt_xdate()
    ax.fmt_xdata = mdates.DateFormatter('%m-%d')
    ax.set_xlabel(xlabel ,fontsize=12,fontweight='bold')
    ax.set_ylabel(ylabel ,fontsize=12,fontweight='bold')
    ax.set_title(title, fontweight='bold',fontsize=15)
    
    
def SelectBetweenDates(df,start,end):
#    '2018-01-01 09:00:00','2018-01-01 11:00:00'
    df_select=df[df["datetime"].between(start,end)]
    df_select=df_select.reset_index(drop=True)
    return df_select
    
#LOAD FULL DATASET
print("Loading DataSets: "+TrainFile+" and " + TestFile)

train_data=pd.read_csv("Data/"+TrainFile)
test_data=pd.read_csv("Data/"+TestFile)

train_data=train_data.drop(['index'], axis=1)
train_data['datetime']=pd.to_datetime(train_data['datetime'])

test_data=test_data.drop(['index'], axis=1)
test_data['datetime']=pd.to_datetime(test_data['datetime'])

#SELECT DEV, TRAIN, TEST SET
TrainSet=train_data.iloc[1:int(1*len(train_data))]
TestSet=test_data
TestSet_Original=TestSet
#MakeCO2Plot(TrainSet)
#MakeCO2Plot(TestSet)

#PREPROCESSING & FEATURE EXTRACTION

print('PREPROCESSING & FEATURE EXTRACTION')
scalingRange= [350,2000]
smoothingMinutes=90
resamplingMinutes=5
originalSampleT=0.5
StdW=5

## MinMax Scaling

TrainSet.co2=(TrainSet.co2-scalingRange[0])/(scalingRange[1]-scalingRange[0])
TestSet.co2=(TestSet.co2-scalingRange[0])/(scalingRange[1]-scalingRange[0])

## Smoothing
TrainSet.co2=TrainSet.co2.rolling(window=int(smoothingMinutes/originalSampleT)).median()
TestSet.co2=TestSet.co2.rolling(window=int(smoothingMinutes/originalSampleT)).median()
### Drop NaNs on the head
TrainSet=TrainSet.reset_index(drop=True).iloc[int(smoothingMinutes/originalSampleT)-2:,]
TestSet=TestSet.reset_index(drop=True).iloc[int(smoothingMinutes/originalSampleT)-2:,]

## Rolling STD Feature Extraction
TrainSet['stdco2']=TrainSet.co2.rolling(window=StdW).std()
TestSet['stdco2']=TestSet.co2.rolling(window=StdW).std()
### Drop NaNs on the head
TrainSet=TrainSet.reset_index(drop=True).iloc[StdW:,]
TestSet=TestSet.reset_index(drop=True).iloc[StdW:,]

## Resampling
TrainSet=TrainSet.iloc[::int(resamplingMinutes/originalSampleT),].reset_index(drop=True)
TestSet=TestSet.iloc[::int(resamplingMinutes/originalSampleT),].reset_index(drop=True)

## Normalized Time Based Features
def ExtractTimeFeatures(df):
    days= [d.dayofweek + d.hour/24+d.minute/1440 for d in df.datetime]
    days=days/max(np.array(days))
    hours=[d.hour + d.minute/60 for d in df.datetime]
    hours=hours/max(np.array(hours))
    df['day']=days
    df['hour']=hours
    return df

TrainSet=ExtractTimeFeatures(TrainSet)
TestSet=ExtractTimeFeatures(TestSet)

## SETTING LEADING AND LAG SERIES
print('SETTING LEADING AND LAG SERIES')
FutureSteps=25
PastSteps=25

def LeadFeatures(df,steps=FutureSteps):
    new=df.copy()
    for i in range(0,steps+1):
        shifted=df.shift(-i) #Dont shift by 0
        for col in df.columns[1:]: #Dont put shifted timestamp column
             new[col+"_P"+str(i)]=shifted[col]
    
    #Remove tail
    new=new.iloc[:-FutureSteps,]
    #Remove head
    new=new.iloc[PastSteps:,]
    
    return new

def LagFeatures(df,steps=PastSteps):
    new=df.copy()
    for i in reversed(range(0,steps+1)):
        shifted=df.shift(i) #Dont shift by 0
        for col in (df.columns[1:]): #Dont put shifted timestamp column
             new[col+"_-"+str(i)]=shifted[col]
    #Remove tail
    new=new.iloc[:-FutureSteps,]
#    #Remove head
    new=new.iloc[PastSteps:,]
    return new


# SEPARATE X and Y
XFutureFeats=[]
XPastFeats=['co2','stdco2_','hour_-','day_-']
YFutureFeats=['co2_']

def PrepareX(Set):
    FutureSet=LeadFeatures(Set)
    PastSet=LagFeatures(Set)
    XPast=pd.concat([PastSet.filter(regex='^'+fname) for fname in XPastFeats],axis=1).reset_index(drop=True)
    try:
        XFuture=pd.concat([FutureSet.filter(regex='^'+fname) for fname in XFutureFeats],axis=1).reset_index(drop=True)
        return (pd.concat([XPast,XFuture],axis=1))
    except:
        return XPast
    

def PrepareY(Set):
    FutureSet=LeadFeatures(Set)
    YFuture=pd.concat([FutureSet.filter(regex='^'+fname) for fname in YFutureFeats],axis=1).reset_index(drop=True)
    return(YFuture)


X_test=PrepareX(TestSet)
Y_test=PrepareY(TestSet)

X_train=PrepareX(TrainSet)
Y_train=PrepareY(TrainSet)

TestSetTrim=TestSet.iloc[:-FutureSteps,]
TestSetTrim=TestSetTrim.iloc[PastSteps:,]


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor


def PrepareX_lstm(X2d):
    hour=X2d.filter(regex='^hour_-') 
    day=X2d.filter(regex= '^day_-' )
    co2=X2d.filter(regex='^co2')
    std=X2d.filter(regex='^std')
    hour_future=X2d.filter(regex='^hour_P')
    day_future=X2d.filter(regex='^day_P')
    
    # reshape training into [samples, timesteps, features]
    X3d=np.array( [np.array(hour),np.array(day),np.array(co2), np.array(std),np.array(hour_future),np.array(day_future)])
    X3d=np.transpose(X3d, (1, 2, 0))
    
    return X3d

# X_train=PrepareX_lstm(X_train)
# X_test=PrepareX_lstm(X_test)


# # design network
# model = Sequential()
# model.add(LSTM(4, input_shape=(X_train.shape[1], X_train.shape[2]),return_sequences=True))
# model.add(BatchNormalization())
# model.add(LSTM(4))
# model.add(Dense(Y_train.shape[1]))
# model.compile(loss='mean_squared_error', optimizer='adam')

# for i in range(5):
#  	model.fit(X_train, Y_train, epochs=1,verbose=1, shuffle=False)
#  	model.reset_states()

print('FITTING REGRESSOR')

regr_rf = RandomForestRegressor(n_estimators=10, max_depth=10,
                                random_state=2)


regr_rf.fit(X_train, Y_train)


# Predict on new data

print('MAKING PREDICTION ON TEST SET')
y_rf = regr_rf.predict(X_test)

TestSetPPM=TestSetTrim.copy()
TestSetPPM['co2']=(scalingRange[1]-scalingRange[0])*TestSet['co2']+scalingRange[0]

y_rf_ppm = (scalingRange[1]-scalingRange[0])*y_rf +scalingRange[0]
Y_test_ppm=(scalingRange[1]-scalingRange[0])*Y_test+scalingRange[0]


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
RMSE=sqrt(mean_squared_error(Y_test_ppm,y_rf_ppm))
MAE=sqrt(mean_absolute_error(Y_test_ppm,y_rf_ppm))
print('RMSE='+str(RMSE)+" PPM")
print('MAE='+str(MAE)+" PPM")


plt.figure()
Difference=abs((Y_test_ppm-y_rf_ppm))
stepErrors=Difference.sum(axis=0)/len(Difference)
np.savetxt("steperrors.csv",stepErrors,delimiter=',')

plt.plot(np.linspace(0,FutureSteps,FutureSteps+1),stepErrors)
plt.title('Prediction Errors as a function of steps')



#PlotResults


plt.figure()
plt.plot(TestSetTrim['datetime'],TestSetPPM['co2'],color='k',linewidth=0.75)

for predX in range(0,len(X_test),50):
    
    if( max(y_rf_ppm[predX])-min(y_rf_ppm[predX]) >10):
        plt.plot(TestSetTrim['datetime'].iloc[predX:predX+1+FutureSteps],y_rf_ppm[predX],linewidth=3,linestyle='--',color='r')
        plt.scatter(TestSetTrim['datetime'].iloc[predX],y_rf_ppm[predX,0],color='k')
        plt.scatter(TestSetTrim['datetime'].iloc[predX+1+FutureSteps],y_rf_ppm[predX,-1],color='k')
    
    
plt.suptitle('CO2 Predictions', fontweight='bold')
plt.title('Predicted Time=2hrs ' + ' InputPastData=2hrs '+' SampleRate=5 minutes')






