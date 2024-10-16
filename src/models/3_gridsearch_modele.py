import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
import joblib

X_train = pd.read_csv('/home/ubuntu/exam_dvc/examen-dvc/data/normalised/X_train_scaled.csv')
y_train = pd.read_csv('/home/ubuntu/exam_dvc/examen-dvc/data/processed/y_train.csv')

model = Ridge()
param_grid = {
    'alpha': [0.1, 1.0, 10.0],  
    'solver': ['auto', 'svd', 'cholesky']
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train.values.ravel())

joblib.dump(grid_search.best_params_, '/home/ubuntu/exam_dvc/examen-dvc/models/best_params.pkl')
