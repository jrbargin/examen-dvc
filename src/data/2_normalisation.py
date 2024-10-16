import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

os.makedirs('/home/ubuntu/exam_dvc/examen-dvc/data/normalised', exist_ok=True)

X_train = pd.read_csv('/home/ubuntu/exam_dvc/examen-dvc/data/processed/X_train.csv')
X_test = pd.read_csv('/home/ubuntu/exam_dvc/examen-dvc/data/processed/X_test.csv')

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv('/home/ubuntu/exam_dvc/examen-dvc/data/normalised/X_train_scaled.csv', index=False)
pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv('/home/ubuntu/exam_dvc/examen-dvc/data/normalised/X_test_scaled.csv', index=False)
