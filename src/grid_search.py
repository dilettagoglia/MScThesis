''' #################################
        K-Fold Cross Validation
    ################################# '''

# import models
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import xgboost as xgb

# import metrics
from sklearn.metrics import make_scorer, r2_score, mean_squared_error, mean_absolute_error

# other imports
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold
import json
from random import randint
from numpy import mean, std
from utilities import run_gs, write_gs_results, write_output_file, remove_prefix
from params import include_sci

def GridSearchConfig(model=None, plot_tree=False):
    '''
    ...

    [ScikitLearn Regression metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics)
    [ScikitLearn RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
    # todo inserire documentazione modelli mancanti
    [XGBoost Documentation](https://xgboost.readthedocs.io/en/stable/index.html)
    '''

    try:

        ''' ###############
             RANDOM FOREST
            ############### '''

        if model == 'RF':
            model_name = 'Random Forest'

            # Define search space
            # to find the best set of parameter setting, first define the parameters' values we want to try
            param_dist = {# "max_depth": [None],
                          # "max_features": ["auto", "sqrt", "log2"], # rimosso: non utile
                          # "min_samples_split": randint(10, 50), # RandomizedSearcCV results: 1:32, 2:20, 3:36
                          # "min_samples_leaf": randint(10, 50), # RandomizedSearcCV results: 1:13, 2:14, 3:19
                          "min_samples_split": [i for i in range(20, 60, 8)],
                          "min_samples_leaf": [i for i in range(2, 28, 2)],
                          "n_estimators": [i for i in range(200, 1000, 200)]  # number of trees in the forest
                          # "bootstrap": [True, False], # use default value (True)
                          # "criterion": ['mse', 'mae'], # use default 'squared_error'
                            # Deprecated since version 1.0: Criterion “mse” was deprecated in v1.0 and will be removed in version 1.2. Use criterion="squared_error" which is equivalent.
                            # Deprecated since version 1.0: Criterion “mae” was deprecated in v1.0 and will be removed in version 1.2. Use criterion="absolute_error" which is equivalent.
                          }

            # Define fixed parameters
            n_iter_search = 50  # number of iterations

            # Define the model
            mod = RandomForestRegressor()

            # if plot_tree==True:
            # tree.export_graphviz(grid_search.best_estimator_.estimators_[k]) #todo

        ''' #########
               SVM
            ######### '''

        if model == 'SVM':
            model_name = 'Epsilon-Support Vector Regression'

            param_dist = {"svr__kernel": ['rbf'], # 'linear', 'poly', 'sigmoid'
                          #"svr__degree": [2, 3, 6],  # considered only if kernel is 'poly' # il degree too high --> overfitting
                          "svr__gamma": ['auto'],  # Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
                          "svr__C": [0.1, 0.3, 0.5, 1, 2, 5], # the smaller C, the more the regularization # Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive. The penalty is a squared l2 penalty.
                          #'svr__coef0': [0, 0.001, 0.0001, 0.01, 0.1, 0.5, 1, 2, 3, 4, 5, 10, 15], # Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’. Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive. The penalty is a squared l2 penalty.
                          "svr__epsilon": [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01], # todo appuntare significato parametro epsilon
                          # It specifies the epsilon-tube within which no penalty is associated in the training loss function with points predicted within a distance epsilon from the actual value.
                          }
            # Due to pipeline: 'To grid search the parameters of the inner dump of such a pipeline you will have to use to prefix with the lowercase version of the class name'

            n_iter_search = 50

            mod = make_pipeline(
                preprocessing.StandardScaler(),
                SVR(),
            )

        ''' #########
               MLP
            ######### '''

        if model == 'MLP':
            model_name = 'Multi-layer Perceptron regressor'
            pref = 'mlpregressor__'

            param_dist = {"hidden_layer_sizes": [(10, 10), (10,), (8, 8), (8,), (4, 4)],
                          "activation": ['tanh'], # tangent chosen due to standard scaler # previously: ['identity','relu'] # default relu
                          "solver": ['adam', 'sgd'],
                          #"alpha": [0.0001], #L2 penalty (regularization term) parameter # default 0.0001
                          "batch_size": [64, 128, 'auto'],
                          "learning_rate": ['constant', 'adaptive'],
                          #"learning_rate_init": [0.1, 0.01, 0.001], # default 0.001
                          "max_iter": [2000, 4000, 8000], # default 200. 'For stochastic solvers (‘sgd’, ‘adam’), note that this determines the number of epochs (how many times each data point will be used), not the number of gradient steps'.
                          "shuffle": [True, False],
                          #"random_state": [None], # default None
                          "momentum": [0.7, 0.8, 0.9],
                          "nesterovs_momentum": [True, False],
                          "early_stopping": [True, False],
                          "random_state": [0] # Multi layer perceptron as well as other neural network architectures suffer from the fact that their corresponding loss functions have numerous local optima. Thus all gradient algorithms are heavily dependent on what initialization is chosen. And rather than seeing this as undesirable you can view the initialization (determined through random_state) as an additional hyperparameter that gives you flexibility.
                          }
            param_dist = {f"{str(pref)}{str(key)}": val for key, val in param_dist.items()}  # due to the pipeline below

            n_iter_search = 50
            mod = make_pipeline(
                preprocessing.StandardScaler(), # todo verificare correttezza: grid search restituisce: ValueError: Input contains NaN, infinity or a value too large for dtype('float64').
                MLPRegressor(),
            )

        ''' #########
               XGB
            ######### '''

        if model == 'XGB':
            model_name = 'XGBoost Regressor'

            param_dist = {'objective':['reg:squarederror', 'reg:squaredlogerror'], # default=reg:squarederror
                          # 'eval_metric':[], # default according to objective
                          'learning_rate': [0.08, 0.05, 0.03], # eta value # default=0.3
                          # 'max_depth': [2, 4, 6, 8], # default=6
                          'subsample': [0, 0.5, 1], # default=1
                          'n_estimators': [350, 500, 750], # deafult=100
                          # 'gamma':[0], # default=0
                          # 'alpha':[0], # L1 regularization # default=0
                          'lambda':[0, 3, 5] # L2 regularization # default=1
                          }

            n_iter_search = 50
            mod = xgb.XGBRegressor()

        return model, mod, param_dist, n_iter_search

    except ValueError:
        print("Insert correct string for 'model' parameter")

def NestedKFoldCV(X, y, model, outer_k=5, inner_k=5, randomized=False):
    '''
    This function implements a nested k-fold cross validation.

    The K-Fold is configured with the number of folds (splits), then the split() function is called,
    passing in the dataset. The results of the split() function are enumerated to give the row indexes
    for the train and test sets for each fold.

    The hyperparameter search is configured to refit a final model with the entire training dataset
    using the best hyperparameters found during the search. This is achieved by setting the “refit”
    argument to True, then retrieving the model via the “best_estimator_” attribute on the search result.
    This model is then used to make predictions on the holdout data from the outer loop and estimate
    the performance.

    Each iteration of the outer cross-validation procedure reports the estimated performance of the best
    performing model (using inner_k-fold cross-validation) and the hyperparameters found to perform the
    best, as well as the R^2 on the holdout dataset.

    NB: Results may vary given the stochastic nature of the algorithm or evaluation procedure,
    or differences in numerical precision. Consider running the example a few times and compare the average outcome.

    [ScikitLearn K-Fold class](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html)
    [Nested versus non-nested cross-validation - from ScikitLearn](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html#sphx-glr-auto-examples-model-selection-plot-nested-cross-validation-iris-py)

    :param X:
    :param y:
    :param outer_k:
    :param inner_k:
    :param randomized:
    :return:
    '''

    search_type = 'Randomized search' if randomized == True else 'Grid search'
    text = f'\n\nConfiguring nested k-fold cross-validation procedure (with {str(outer_k)} outer and {str(inner_k)} inner folds) ...'

    cv_outer = KFold(n_splits=outer_k) # outer-loop of the nested-cross validation procedure      # ValueError: Setting a random_state has no effect since shuffle is False. You should leave random_state to its default (None), or set shuffle=True.
    # todo provare shuffle true
    outer_results = list() # enumerate splits
    best_params = dict()
    fold_count = 0

    for train, test in cv_outer.split(X):
        # The .split() function generate indices to split data into training and test set
        X_train, X_test = X.iloc[train, :], X.iloc[test, :]
        y_train, y_test = y[train], y[test]
        cv_inner = KFold(n_splits=inner_k, shuffle=True, random_state=1) # the train dataset defined by the outer loop is used as the dataset for the inner loop.
        model_name, mod, param_dist, n_iter_search = model

        # run the search and write results
        scoring = make_scorer(r2_score)  # scoring system
        grid_search = run_gs(X_train, y_train, mod, param_dist, n_iter_search, scoring, randomized=randomized,
                             cv=cv_inner, refit=True)
        write_gs_results(grid_search, search_type, model_name)

        best_model = grid_search.best_estimator_ # get the best performing model fit on the whole training set

        # evaluate model on the hold out dataset
        test_pred = best_model.predict(X_test)
        # evaluate the model
        r2 = r2_score(y_test, test_pred)
        mse = mean_squared_error(y_test, test_pred) # default: squared=True
        rmse = mean_squared_error(y_test, test_pred, squared=False)
        mae = mean_absolute_error(y_test, test_pred)
        # store the result
        outer_results.append(r2)
        best_params.update({r2 : grid_search.best_params_}) # append parameters as dictionary values with r2 as key

        fold_count+=1

        # report progress
        text += f'\n\nInner fold #{str(fold_count)}\n>r2=%.3f, best=%.5f, cfg=%s' % (r2, grid_search.best_score_, grid_search.best_params_)
        text += f'\nMSE:\t{str(mse)}\nRMSE:\t{str(rmse)}\nMAE:\t{str(mae)}'
        # print("Best Score: {}".format(grid_search.best_score_))

    # summarize the estimated performance of the model
    text += f'\n{str(search_type)} with Nested K-fold Cross Validation for {str(model_name)} model completed\n\n' \
           f'Mean R2: %.3f (%.3f)' % (mean(outer_results), std(outer_results))
    write_output_file(text, 'nested_kfold_results')

    res = best_params[max(best_params)] # FINAL RESULT

    subfolder = '' if include_sci else 'no_sci_models/'

    # save
    json.dump(res, open(f"../dump/{str(subfolder)}{str(model_name)}.json", "w"))

    return res # returns the best configuration of parameters by selecting the value (config) with the highest key (r2) in the dictionary