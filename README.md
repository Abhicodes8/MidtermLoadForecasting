# Midterm Load Forecasting
This repository provides a comprehensive solution for mid- to long-term electricity load forecasting using machine learning models. The primary script, load_forecasting.py, is designed to forecast hourly energy consumption in the SCENT region (ERCOT, Texas) using weather, calendar, and historical load features. It includes built-in data preprocessing, model training, evaluation, and prediction generation.

## What is this? What is the objective?
The goal of this project is to improve the accuracy and interpretability of electricity load forecasts for grid planning and energy resource management. By leveraging advanced machine learning models (XGBoost, LSTM, etc) and SHAP-based explainability, this pipeline enables users to:
1. Predict electricity demand from several hours up to months in advance.
2. Analyze feature importance and temporal patterns as well as maintaing explainability at the same time. 

## Prerequisites.
You will need Python 3 and the following libraries:

For general use:

pip install pandas numpy matplotlib scikit-learn xgboost shap openpyxl
For LSTM models:

pip install tensorflow
If you're using Google Colab, most dependencies are pre-installed. Any missing ones can be installed using pip inside the notebook.



## How to use this program:

## 3.1 In Google Colab (Recommended)
Open the script in Colab.

Run the first cell to upload your input Excel files.

Follow the prompts to:

Upload weather_data.xlsx (with columns like timestamp, tavg, tmin, tmax, etc.)

Upload load_data.xlsx (with columns like timestamp, load_MW)

The script will:

1. Clean and merge the datasets

2. Perform feature engineering

3. Train the model and make predictions

4. Output metrics and generate visualizations

## 3.2 Locally
If you prefer to run it locally:

python load_forecasting_models.py
You’ll be prompted to upload the Excel files through the console or file picker interface, depending on how you adapt the script.

## 4. Output
After execution, the script will generate:

1. forecast.csv — predicted vs actual load values

2. forecast_plot.png — a visualization of model predictions

3. shap_summary.png (optional) — SHAP feature importance summary

4. Printed performance metrics: MAPE, RMSE, etc.

## 5. Customization
Inside the script, you can modify:

1. The forecast horizon (e.g., 24 hours ahead, 1 month ahead)

2. The model type (e.g., XGBoost, LSTM)

3. Whether to enable SHAP explainability

4. Which features are included (e.g., temperature lags, rolling means, holiday flags)


## 6. Building Paper 

