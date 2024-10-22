import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
import joblib
import yaml

X_train = pd.read_csv('/home/ubuntu/exam_dvc/examen-dvc/data/normalised/X_train_scaled.csv')
y_train = pd.read_csv('/home/ubuntu/exam_dvc/examen-dvc/data/processed/y_train.csv')

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)
    
model = Ridge()
param_grid = {'alpha': params['grid_search']['alpha'],'solver': params['grid_search']['solver']}

grid_search = GridSearchCV(model, param_grid, cv=params['grid_search']['cv'], scoring=params['grid_search']['scoring'])
grid_search.fit(X_train, y_train.values.ravel())

joblib.dump(grid_search.best_params_, '/home/ubuntu/exam_dvc/examen-dvc/models/best_params.pkl')
