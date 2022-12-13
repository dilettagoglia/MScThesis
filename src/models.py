import json
import time
from datetime import datetime

import matplotlib
from matplotlib import pyplot as plt, pyplot
#matplotlib.use('Agg')
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from utilities import impurity_feature_importance, permutation_feature_importance, write_output_file
from params import include_sci, feature_selection
from regression import y
from grid_search import *


subfolder = '' if include_sci else 'no_sci_models/'

def LoadBestModels(X, Y, model_name, kfold=True, randomized=True):
    if kfold == True:
        best_model_config = NestedKFoldCV(X, Y, GridSearchConfig(model=model_name), randomized=randomized)
    else:
        best_model_config = json.load(open(f"../dump/{str(subfolder)}{str(model_name)}.json", "r"))  # load json file

    # handle prefix in parameters (only for models with pipelines due to data normalization)
    a = (model_name == 'MLP')
    b = (model_name == 'SVM')

    if a or b:
        # todo: fix this, horrible syntax :(
        for key in best_model_config.copy():
            # new_key = remove_prefix(key, '__') # remove everything before a character in a string # NOT WORKING
            if a:
                new_key = key[14:]  # remove firsts 5 chars on left
            if b:
                new_key = key[5:]  # remove firsts 5 chars on left

            best_model_config[new_key] = best_model_config.pop(key)  # replace key in dict


    print(f'\n{str(model_name)} best configuration: ', best_model_config)

    return best_model_config

def ModelRun(model, mod_name, X_train, y_train, X_test, y_test, plot=True, **kwargs):
    '''

    :param X_train:
    :param y_train:
    :param X_val:
    :param y_val:
    :param plot:
    :param kwargs:
    :return:
    '''

    # scores = pd.DataFrame(columns=['date', 'model', 'r2_tr', 'mse_tr', 'rmse_tr', 'mae_tr', 'r2_ts', 'mse_ts', 'rmse_ts', 'mae_ts']) # uncomment this line to reset the csv file

    ''' ############
            FIT
        ############ '''
    print(f'\n\n{str(mod_name)} best model running ...')
    print(f'{str(mod_name)} fit ...')

    model.fit(X_train, y_train)

    ''' ########################
           FEATURE IMPORTANCE
        ######################## '''
    print('Computing feature importance ...')
    sns.set_theme()
    permutation_feature_importance(model, X_test, y_test, mod_name)

    # if mod_name == 'RF':
        # impurity_feature_importance(model, X_train, mod_name) # per rf # temporaneamente rimosso

    ''' if mod_name == 'XGB':
        permutation_feature_importance(model, X_test, y_test, mod_name)
        xgb.plot_importance(model.get_booster())
        plt.savefig(f'../img/{str(mod_name)}_feature_importance')
        plt.show(block=True)  # plt.show must be after plt.savefig
        
        # plot the output tree via matplotlib, specifying the ordinal number of the target tree
        #xgb.plot_tree(xgb_model, num_trees=xgb_model.best_iteration)
        # converts the target tree to a graphviz instance
        fig, ax = plt.subplots(figsize=(30, 30))
        xgb.to_graphviz(xgb_model, num_trees=xgb_model.best_iteration, ax=ax)
        plt.savefig(f'../img/xgb_tree')
        plt.show()
    '''


    ''' ############
          PREDICT
        ############ '''
    start_time = time.time()
    print(f'{str(mod_name)} predict ...')
    train_pred = model.predict(X_train)
    #print(f'\n\n{str(mod_name).upper()} prediction on TR set: ', train_pred)
    test_pred = model.predict(X_test)
    #print(f'\n\n{str(mod_name).upper()} prediction on TS set: ', test_pred)
    elapsed_time = time.time() - start_time


    ''' ############
          METRICS
        ############ '''
    ds = ['TR', 'TS']
    pred = [train_pred, test_pred]
    y_vectors = [y_train, y_test]

    text = f'Model specifications:\n{str(model)}'
    caption = ''

    new_row = [str(datetime.now()), mod_name]

    for i in range(2):

            print(f'Computing metrics for {str(mod_name)} predictions on {str(ds[i])} set')
            r2 = r2_score(y_vectors[i], pred[i])
            mse = mean_squared_error(y_vectors[i], pred[i])  # default: squared=True
            rmse = mean_squared_error(y_vectors[i], pred[i], squared=False)
            mae = mean_absolute_error(y_vectors[i], pred[i])

            new_row.extend((r2, mse, rmse, mae))

            metric_text = f'\n\n{str(mod_name)} {str(ds[i])} set predictions outcome' \
                          f'\n\nR2:\t%.4f\nMSE:\t%.8f\nRMSE:\t%.8f\nMAE:\t%.8f' % (r2, mse, rmse, mae)
            text += metric_text
            caption += metric_text
            text += f'\n\nElapsed time to compute predictions: {elapsed_time:.6f} seconds'
            write_output_file(text, 'models_results')

    ''' ############
          PLOTS
        ############ '''
    if plot==True:

        ''' ############
              SCATTER
            ############ '''

        print('Producing plots ...')
        fig, ax = plt.subplots(figsize=(18, 18))
        plt.ylim([0.000001, 1])
        plt.xlim([0.000001, 1])

        if include_sci:
            sci = X_test['sci_2020']
            c = sci.apply(np.log)
            lab = 'SCI 2020 - Log'
            suffix=''
        else:
            share_cont = X_test['share_cont']
            c = share_cont
            lab = 'Sharing continent'
            suffix='_noSCI'

        plt.scatter(y_test, test_pred, marker='.', c=c, s=300, lw=0, alpha=0.7, edgecolor='k', cmap='viridis')

        plt.title(f'{str(mod_name).upper()} predictions of: {str(y)}', fontsize=22) # y is imported from src>regression.py
        plt.xlabel(f"True values - Log", fontsize=20)
        plt.ylabel(f"Predicted values - Log", fontsize=20)
        plt.yscale('log')
        plt.xscale('log')

        #fig.text(1, 1, metric_text)
        plt.figtext(0.5, 0.01, caption, wrap=True, horizontalalignment='left', fontsize=18)

        plt.colorbar(label=lab)
        plt.plot([0, plt.xlim()[1]], [0, plt.xlim()[1]]) # bisettrice

        plt.savefig(f'../img/{str(mod_name)}_scatterplot{str(suffix)}')
        plt.show()

        ''' ###################
              LEARNING CURVES
            ################### '''

        if mod_name=='MLP':
            '''
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                ax1.plot(epochs, val_acc, label='Validation', c='b')
                ax1.set_title('Accuracy')
                ax1.set_xlabel('Epochs')
                ax1.set_ylabel('Accuracy')
                ax2.plot(epochs, loss, label='Training', c='r')
                ax2.plot(epochs, val_loss, label='Validation', c='b')
                ax2.set_title('Loss')
                ax2.set_xlabel('Epochs')
                ax2.set_ylabel('Loss')

                ax1.legend()
                ax2.legend()
                plt.show()
                plt.savefig(f'../img/{str(out_file_name)}.png')
                '''

            plt.plot(model[1].validation_scores_)
            plt.show()
            # todo implementare savefig
            # todo creare ax

    ''' ###########
          SCORES
        ########### '''
    suffix = '' if include_sci else '_noSCI' 
    if feature_selection == False:
        suffix += '_no_feat_sel'
    scores = pd.read_csv(f'../dump/scores{str(suffix)}.csv', sep=",")
    scores.loc[len(scores)] = new_row
    print('*' * 20, scores)
    scores.to_csv(f'../dump/scores{str(suffix)}.csv', index=False, encoding='utf-8')


