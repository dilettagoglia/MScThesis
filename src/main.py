''' #################
        Main File
    ################# '''

# import models
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import xgboost as xgb

# other imports
import json
from regression import X_orig, y, selected_variables
from utilities import dataset_split
from models import *
from params import feature_selection
import pandas as pd

#X_orig.reset_index(inplace=True, drop=True) # RIMOSSO: l'indice deve rimanere quello originale per risalire alle coppie di paesi cui i dati si riferiscono

''' DATA '''
print('\nX matrix:')
print(X_orig.info())
print('\nVariable to predict:', y)
Y = X_orig[y]
print('\nY vector:\n', Y)
X_orig.drop(y, axis=1, inplace=True) # remove label to predict
selected_variables.remove("intercept")
#selected_variables.remove("const")
print('\nVariables selected by Linear Regression model:', selected_variables)

''' MATRIX OF SELECTED VARIABLES '''
X = X_orig[selected_variables]
X.to_csv('../dump/independent_vars_matrix.csv', index = True, encoding='utf-8') # externally save X matrix
Y.to_csv('../dump/dependent_var.csv', index = True, encoding='utf-8') # externally save Y vector

if feature_selection == False: # attention: set this var to True in 'params.py' to RUN EVERYTHING WITHOUT FEATURE SELECTION PHASE
    X = X_orig

print('\nMatrix of selected variables:')
print(X.info())

''' TRAIN / TEST SPLIT '''
print('TR / VAL / TS splitting ...')
X_train, y_train, X_test, y_test = dataset_split(X, Y, valid=False)
print('\nCouples of countries in training set: ', X_train.index) # todo temporary prints
print('Couples of countries in test set: ', X_test.index)
print('Dataset splitted ✓')

''' LOAD OR COMPUTE THE BEST PARAMETER CONFIGURATIONS '''
configurations = []
models = {'RF': False, 'SVM': False, 'MLP': False, 'XGB': False} # dictionary: key = model name, value = whether to execute the kfold for that model
for mod, kf in models.items():
    configurations.append(LoadBestModels(X, Y, mod, kfold=kf, randomized=False))

# external log
txt = 'WITH SCI' if include_sci else 'WITHOUT SCI'
write_output_file(txt, 'models_results')


''' MODEL 1: RANDOM FOREST '''
ModelRun(RandomForestRegressor(**configurations[0]), 'RF', X_train, y_train, X_test, y_test, plot=True)

''' MODEL 2: SUPPORT VECTOR MACHINE FOR REGRESSION (SVR) '''
ModelRun(make_pipeline(
    preprocessing.StandardScaler(),
    SVR(**configurations[1])
    ), 'SVM', X_train, y_train, X_test, y_test, plot=True)
# {"C": 12, "epsilon": 0.0005, "gamma": "auto", "kernel": "rbf"}

''' MODEL 3: MULTI-LAYER PERCEPTRON '''
ModelRun(make_pipeline(
    preprocessing.StandardScaler(),
    MLPRegressor(**configurations[2])
    ), 'MLP', X_train, y_train, X_test, y_test, plot=True)

''' MODEL 4: XGBOOST '''
xgb_model = xgb.XGBRegressor(**configurations[3])
ModelRun(xgb_model, 'XGB', X_train, y_train, X_test, y_test, plot=True)

