# Midterm Load Forecasting
This repository provides a comprehensive solution for mid- to long-term electricity load forecasting using machine learning models. The primary script, load_forecasting.py, is designed to forecast hourly energy consumption in the SCENT region (ERCOT, Texas) using weather, calendar, and historical load features. It includes built-in data preprocessing, model training, evaluation, and prediction generation.

## Objective
The goal of this project is to improve the accuracy and interpretability of electricity load forecasts for grid planning and energy resource management. By leveraging advanced machine learning models (XGBoost, LSTM, etc) and SHAP-based explainability, this pipeline enables users to:

Predict electricity demand from several hours up to months in advance.

Analyze feature importance and temporal patterns as well as maintaing explainability at the same time. 

## Setup
1. Clone the Repository
cd load-forecasting
2. Set Up a Virtual Environment 
3. Install Dependencies
pip install -r requirements.txt

## Program usage

### 1. Prepare the Data
Place raw hourly load and weather XCEL files inside when prompted to do so. Files must include:
Contains columns like timestamp, load_MW

weather.xcel: with columns like timestamp, tavg, tmin, tmax, etc.

The script will handle cleaning, merging, feature engineering, and formatting.

### 2. Run the Forecasting Pipeline

python load_forecasting_models.py
This will:
1. Preprocess data (handling holidays, lags, rolling stats, feature encoding)
2. Train the model(s)
3. Evaluate using metrics like MAPE, RMSE, and MAE
4. Save output forecasts and print them out.

Optionally generate SHAP plots and save them

## Customize Settings
You can modify parameters inside load_forecasting.py:

Forecast horizon (1 day to 6 months)

Model type (XGBoost, LSTM, LinearRegression)

Feature inclusion (temperature lags, rolling mean, holidays)

## Dependencies
Key Libraries Include:
1. pandas
2. numpy
3. scikit-learn
4. xgboost
5. matplotlib
6. shap
7. tensorflow (if using LSTM)
