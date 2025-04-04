# Import Data Manipulation Libraries
import numpy as np
import pandas as pd

# import Data Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Import Data Filter Warnings
import warnings
warnings.filterwarnings("ignore")

# Import Data Logging Libraries
import logging
logging.basicConfig(level=logging.INFO,
                    filename="model.log",
                    filemode="w",
                    format="%(asctime)s - %(levelname)s - %(message)s"
                    )

# Load the Dataset

url = "https://raw.githubusercontent.com/Digraskarpratik/CementCompositeStrength_PredicationModel/refs/heads/main/concrete_data.csv"

df = pd.read_csv(url,sep= ",")

df.sample(frac = 1)

df['Composite_Ratio'] = 1/(df['cement'] + df['superplasticizer'] + df['blast_furnace_slag']) / (df['water'])
df["cement_to_water_ratio"] = 1/(df["cement"] / df["water"])

from sklearn.model_selection import train_test_split

X = df.drop(columns="concrete_compressive_strength", axis=1)
y = df["concrete_compressive_strength"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state= 42)

# Using Scaling Technique

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.ensemble import RandomForestRegressor

RF = RandomForestRegressor()
RF.fit(X_train, y_train)

y_pred_RF = RF.predict(X_test)

r2_score_RF = r2_score(y_test, y_pred_RF)

print(f'The R2 Score for Random Forest Regressor :- {r2_score_RF * 100}%')

# Using Boosting Algorithm
import xgboost as xgb

xgb = xgb.XGBRegressor()

xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)

r2_score_xgb = r2_score(y_test, y_pred_xgb)

print(f'The R2 Score for XGBoost :- {r2_score_xgb * 100}%')

# using adaboost algorithm
from sklearn.ensemble import AdaBoostRegressor

ADA = AdaBoostRegressor()

ADA.fit(X_train, y_train)
y_pred_ADA = ADA.predict(X_test)

r2_score_ADA = r2_score(y_test, y_pred_ADA)

print(f'The R2 Score for AdaBoost :- {r2_score_ADA * 100}%')