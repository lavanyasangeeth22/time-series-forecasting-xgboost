# Time Series Forecasting with XGBoost — Daily Minimum Temperatures

This project builds a time series forecasting model to predict **daily minimum temperatures** using **XGBoost regression**, based on the [Daily Minimum Temperatures in Melbourne](https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv) dataset. It covers data loading, date-based and lag/rolling feature engineering, a chronological train/test split, model training with early stopping, evaluation (RMSE), and feature importance analysis.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Step-by-Step Walkthrough](#step-by-step-walkthrough)
  - [Step 1: Import Libraries & Load Data](#step-1-import-libraries--load-data)
  - [Step 2: Visualize the Raw Time Series](#step-2-visualize-the-raw-time-series)
  - [Step 3: Feature Engineering](#step-3-feature-engineering)
  - [Step 4: Chronological Train/Test Split](#step-4-chronological-traintest-split)
  - [Step 5: Train the XGBoost Model](#step-5-train-the-xgboost-model)
  - [Step 6: Evaluate & Visualize Predictions](#step-6-evaluate--visualize-predictions)
  - [Step 7: Feature Importance](#step-7-feature-importance)
- [Results](#results)
- [How to Run](#how-to-run)
- [License](#license)

---



## Overview

Forecasting a continuous value over time (temperature, sales, energy demand, etc.) is a common regression problem. Rather than using a classical statistical model (like ARIMA), this project treats forecasting as a **supervised learning problem**: calendar-based and lag-based features are engineered from the date index, and a gradient-boosted tree model (XGBoost) learns to predict the target from those features.

## Dataset

- **Source:** [Daily Minimum Temperatures in Melbourne](https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv) (Jason Brownlee's public dataset repository)
- **Range:** 1981–1990 (10 years of daily readings)
- **Columns:** `Date`, `Temp` (daily minimum temperature in °C)



## Requirements

```bash
pip install pandas numpy matplotlib seaborn xgboost scikit-learn
```

---



## Step-by-Step Walkthrough



### Step 1: Import Libraries & Load Data

We import the core data science stack plus `xgboost` for modeling, set a consistent plotting style, and load the dataset directly from a raw GitHub CSV URL. The `Date` column is parsed as a datetime index, the value column is renamed to `Temp`, and any non-numeric or missing values are cleaned out.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_squared_error

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 5)

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
print("Loading dataset...")

df = pd.read_csv(url)
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')
df.columns = ['Temp']

df['Temp'] = pd.to_numeric(df['Temp'], errors='coerce')
df = df.dropna().sort_index()
df.head()
```

**Output:**

```
Loading dataset...
```


| Date       | Temp |
| ---------- | ---- |
| 1981-01-01 | 20.7 |
| 1981-01-02 | 17.9 |
| 1981-01-03 | 18.8 |
| 1981-01-04 | 14.6 |
| 1981-01-05 | 15.8 |




### Step 2: Visualize the Raw Time Series

Plotting the full 10-year series shows a clear **yearly seasonal pattern** — temperatures cycle up and down every year, which is exactly the kind of structure calendar-based features (month, day-of-year, etc.) are meant to capture.

```python
plt.figure(figsize=(12, 4))
plt.plot(df.index, df['Temp'], color='navy', alpha=0.7, label='Daily Min Temp')
plt.title('raw-data-plot.png')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.tight_layout()
plt.show()
```

**Output:**

Daily Minimum Temperatures (Melbourne)

### Step 3: Feature Engineering

A `create_features()` function derives predictors from the datetime index:

**Calendar features:**


| Feature      | Description                            |
| ------------ | -------------------------------------- |
| `dayofweek`  | Day of the week (0–6)                  |
| `quarter`    | Calendar quarter (1–4)                 |
| `month`      | Month (1–12)                           |
| `year`       | Year                                   |
| `dayofyear`  | Day number within the year (1–365/366) |
| `dayofmonth` | Day number within the month            |


**Lag & rolling features** (capture short-term momentum in the series):


| Feature          | Description                                                              |
| ---------------- | ------------------------------------------------------------------------ |
| `lag_1`          | Temperature 1 day ago                                                    |
| `lag_7`          | Temperature 7 days ago                                                   |
| `rolling_mean_7` | Mean of the previous 7 days (shifted so it doesn't leak the current day) |


Rows with `NaN` values created by the shift/rolling operations (the first few rows of the dataset, where there's no prior history) are dropped.

```python
def create_features(dataframe):
    df_feat = dataframe.copy()
    
    # Calendar features
    df_feat['dayofweek'] = df_feat.index.dayofweek
    df_feat['quarter'] = df_feat.index.quarter
    df_feat['month'] = df_feat.index.month
    df_feat['year'] = df_feat.index.year
    df_feat['dayofyear'] = df_feat.index.dayofyear
    df_feat['dayofmonth'] = df_feat.index.day
    
    # Lag & Rolling features (improves tracking of short-term weather trends)
    df_feat['lag_1'] = df_feat['Temp'].shift(1)
    df_feat['lag_7'] = df_feat['Temp'].shift(7)
    df_feat['rolling_mean_7'] = df_feat['Temp'].shift(1).rolling(window=7).mean()
    
    return df_feat

df_features = create_features(df)
df_features = df_features.dropna()
df_features.head()
```

**Output:**


| Date       | Temp | dayofweek | quarter | month | year | dayofyear | dayofmonth | lag_1 | lag_7 | rolling_mean_7 |
| ---------- | ---- | --------- | ------- | ----- | ---- | --------- | ---------- | ----- | ----- | -------------- |
| 1981-01-08 | 17.4 | 3         | 1       | 1     | 1981 | 8         | 8          | 15.8  | 20.7  | 17.057143      |
| 1981-01-09 | 21.8 | 4         | 1       | 1     | 1981 | 9         | 9          | 17.4  | 17.9  | 16.585714      |
| 1981-01-10 | 20.0 | 5         | 1       | 1     | 1981 | 10        | 10         | 21.8  | 18.8  | 17.142857      |
| 1981-01-11 | 16.2 | 6         | 1       | 1     | 1981 | 11        | 11         | 20.0  | 14.6  | 17.314286      |
| 1981-01-12 | 13.3 | 0         | 1       | 1     | 1981 | 12        | 12         | 16.2  | 15.8  | 17.542857      |




### Step 4: Chronological Train/Test Split

Because this is time series data, the split must be done **chronologically** (not randomly) to avoid leaking future information into training. Everything before **1988-01-01** becomes the training set; everything from that date onward becomes the test set.

```python
split_date = '1988-01-01'
train = df_features.loc[df_features.index < split_date]
test = df_features.loc[df_features.index >= split_date]

FEATURES = ['dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'dayofmonth', 'lag_1', 'lag_7', 'rolling_mean_7']
TARGET = 'Temp'

X_train, y_train = train[FEATURES], train[TARGET]
X_test, y_test = test[FEATURES], test[TARGET]
print(f"Training set: {len(train)} rows | Testing set: {len(test)} rows")
```

**Output:**

```
Training set: 2548 rows | Testing set: 1095 rows
```



### Step 5: Train the XGBoost Model

An `XGBRegressor` is trained with **1,000 estimators**, a low learning rate (0.01), and **early stopping** (stops if the test RMSE doesn't improve for 50 rounds), which helps prevent overfitting. Both the train and test sets are passed as evaluation sets so RMSE can be tracked on both during training.

```python
reg = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=5,
    early_stopping_rounds=50,
    random_state=42
)
reg.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=100
)
```

**Output (training log, every 100 rounds):**

```
[0]	validation_0-rmse:4.05331	validation_1-rmse:4.02667
[100]	validation_0-rmse:2.60558	validation_1-rmse:2.63360
[200]	validation_0-rmse:2.24242	validation_1-rmse:2.35682
[300]	validation_0-rmse:2.11925	validation_1-rmse:2.32160
[400]	validation_0-rmse:2.05192	validation_1-rmse:2.31474
[439]	validation_0-rmse:2.03362	validation_1-rmse:2.31533
```

**Insight:** Training stopped early around round 439 (rather than running the full 1,000) because the test RMSE (`validation_1-rmse`) stopped improving — early stopping did its job, preventing the model from continuing to fit noise in the training set.

### Step 6: Evaluate & Visualize Predictions

Predictions are generated for the test set, and **RMSE** (Root Mean Squared Error) is calculated to quantify how far off, on average, the predictions are from the actual temperatures (in °C). The results are then plotted alongside the training and test actuals for a visual comparison.

```python
test = test.copy()
test['Prediction'] = reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(test[TARGET], test['Prediction']))
print(f"\nRoot Mean Squared Error (RMSE): {rmse:.4f}")

plt.figure(figsize=(12, 5))
plt.plot(train.index, train[TARGET], label='Train Actual', color='gray', alpha=0.5)
plt.plot(test.index, test[TARGET], label='Test Actual', color='blue')
plt.plot(test.index, test['Prediction'], label='XGBoost Prediction', color='orange', linestyle='--')
plt.title('forecast-results.png')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.tight_layout()
plt.show()
```

**Output:**

```
Root Mean Squared Error (RMSE): 2.3141
```

XGBoost Temperature Forecast: Actual vs Predicted

**Insight:** The predicted curve (orange, dashed) tracks the actual test data (blue) closely, correctly capturing the seasonal rise-and-fall pattern across each of the three test years (1988–1990), with an average error of about **±2.3°C**.

### Step 7: Feature Importance

Finally, we inspect which features the model relied on most to make its predictions.

```python
importance_df = pd.DataFrame({
    'Feature': FEATURES,
    'Importance': reg.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(8, 4))
sns.barplot(
    data=importance_df, 
    x='Importance', 
    y='Feature', 
    hue='Feature', 
    palette='viridis', 
    legend=False
)
plt.title('feature-importance.png')
plt.tight_layout()
plt.show()
```

**Output:**

XGBoost Feature Importance

**Insight:** `rolling_mean_7` and `lag_1` are by far the most influential features — together they account for the vast majority of the model's decision-making. This makes intuitive sense for weather data: **yesterday's temperature and the recent 7-day trend are strong predictors of tomorrow's temperature**, far more so than the calendar position alone (`dayofyear`, `quarter`, `month`, etc.), which contribute comparatively little.

---



## Results


| Metric           | Value                              |
| ---------------- | ---------------------------------- |
| Model            | XGBoost (`XGBRegressor`)           |
| Train/Test Split | Chronological, split at 1988-01-01 |
| Training Rows    | 2,548                              |
| Test Rows        | 1,095                              |
| Best Iteration   | ~439 (early stopped)               |
| **Test RMSE**    | **2.3141 °C**                      |
| Top Feature      | `rolling_mean_7`                   |




## How to Run

1. Clone this repository:
  ```bash
   git clone https://github.com/lavanyasangeeth22/time-series-forecasting-xgboost
   cd time-series-forecasting-xgboost
  ```
2. Install dependencies:
  ```bash
   pip install pandas numpy matplotlib seaborn xgboost scikit-learn
  ```
3. Run the notebook/script (e.g., in Jupyter, VS Code, or Google Colab):
  ```bash
   jupyter notebook temperature_forecasting.ipynb
  ```



## License

This project is for educational purposes, based on the publicly available [Daily Minimum Temperatures dataset](https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv) (Jason Brownlee, Machine Learning Mastery).
