# Time Series Forecaster
A template for preparing time series data into an elegant X Y problem. Initially designed to handle indoor CO2 data, extract a few features, make leads and lags, and finally fitting it to a Random Forest Regressor. 

I shall be making it more generic in the commits to come.

* ## Complete Data Pipeline

![pipeline](/Images/Pipeline.PNG)

* ## Extracted Feature #1 - Window Standard Deviation: In addition to smoothed data, a windowed standard deviation metric was generated to detect sharp changes in the measurement. 
 
![f1](/Images/Features_WindowStd.PNG)
 
* ## Extracted Feature #2 and #3 - Time Based features: As a result of our data exploration, 2 time based features were generated from hour of the day, and day of the week as these parameters are potentially good inputs to the cyclic nature of the data. 

![f2](/Images/Features_Times.PNG)

* ## Sample Results: Some selected predictions on the test set are shown below. For brevity only every 50th prediction is plotted

![f3](/Images/SampleResults.PNG)
