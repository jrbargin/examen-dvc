import pandas as pd
from sklearn.model_selection import train_test_split
import os

os.makedirs('/home/ubuntu/exam_dvc/examen-dvc/data/processed', exist_ok=True)

df = pd.read_csv('/home/ubuntu/exam_dvc/examen-dvc/data/raw/raw.csv')
X = df.drop(['date','silica_concentrate'],axis=1)
y = df['silica_concentrate']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_train.to_csv('/home/ubuntu/exam_dvc/examen-dvc/data/processed/X_train.csv', index=False)
X_test.to_csv('/home/ubuntu/exam_dvc/examen-dvc/data/processed/X_test.csv', index=False)
y_train.to_csv('/home/ubuntu/exam_dvc/examen-dvc/data/processed/y_train.csv', index=False)
y_test.to_csv('/home/ubuntu/exam_dvc/examen-dvc/data/processed/y_test.csv', index=False)