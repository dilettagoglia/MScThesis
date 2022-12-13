'''
######################
    USER INTERFACE
######################
'''
'''
WHICH INDICATOR YOU WANT TO PREDICT?

                INSERT VALUE HERE   |   CHOOSE BETWEEN THE FOLLOWING: '''
indicator =     'BMI'                   # Unidirectional indicators: 'UMI', 'UMP', Bidirectional indicators: 'BMI', BMP'
by =            'res'                   # 'res' or 'cit' for migration flows by residence or citizenship respectively
source =        'ESTAT'                 # 'ESTAT' for Eurostat data or 'UN' for United Nations data
year =          2019                    # for the complete list of available years please see ... #todo riferimenti a paper per range di valori (es. anni)
sex =           'T'                     # 'F', 'M', 'T' for Female, Male or Total population respectively
age_group =     'T'                     # todo finire parte testuale


'''
##################################
    PLEASE IGNORE WHAT FOLLOWS
##################################
'''

# DATA SOURCE
online_path = 'https://zenodo.org/record/6493325/files/mimi_dataset_v2.csv?download=1'
offline_path = '../data/mimi_dataset_v2.csv'
run_ols = True
include_sci = True
feature_selection = True

# SETTING PARAMETERS
def choose_params(returndf=True):
    ''' Returns the corresponding variables to use (dataframe, X and y) '''

    df_str = f'mimi_not_null_{str(source)}_{str(indicator)}_{str(by)}' # DF (e.g, mimi_not_null_ESTAT_BMI_res)
    y = f'{str(source)}_flow_{str(indicator)}_{str(year)}_{str(sex)}_{str(age_group)}_{str(by)}' # Y (e.g., 'ESTAT_flow_BMI_2019_T_T_res')

    if returndf==True:
        return df_str, y

    else:
        return y


