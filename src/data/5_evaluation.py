import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json


X_test = pd.read_csv('data/processed/X_test_scaled.csv')
y_test = pd.read_csv('data/processed/y_test.csv')
model = joblib.load('src/models/model.pkl')

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

metrics = {'mse': mse, 'r2': r2}
with open('metrics/scores.json', 'w') as f:
    json.dump(metrics, f)

pd.DataFrame(y_pred, columns=['predictions']).to_csv('data/predictions.csv', index=False)
