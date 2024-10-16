import pandas as pd
from sklearn.linear_model import Ridge
import joblib

X_train = pd.read_csv('/home/ubuntu/exam_dvc/examen-dvc/data/processed/X_train_scaled.csv')
y_train = pd.read_csv('/home/ubuntu/exam_dvc/examen-dvc/data/processed/y_train.csv')
best_params = joblib.load('/home/ubuntu/exam_dvc/examen-dvc/models/best_params.pkl')

model = Ridge(**best_params)
model.fit(X_train, y_train.values.ravel())

joblib.dump(model, '/home/ubuntu/exam_dvc/examen-dvc/models/model.pkl')