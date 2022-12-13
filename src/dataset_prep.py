from utilities import bool_vars, compute_indices, num_vars

def prepare_dataset(mimi):

    mimi = bool_vars(mimi)

    # controlli a campione
    #print('\nNeighbors check:\n',mimi[mimi['neighbors']==1][['origin_name', 'destination_name', 'neighbors']].head(50))
    #print('\nReligion check:\n', mimi[mimi['share_rel']==1][['origin_name', 'destination_name', 'share_rel']][1150:1200])
    #print('\nLanguage check:\n', mimi[['origin_languages','destination_languages', 'share_lang']][500:540])

    mimi, mimi_tot_indices = compute_indices(mimi)

    #controlli a campione
    #print('\nID check:\n', mimi[['id','from_to', 'stocks_BMP_2020_T_T', 'UN_flow_BMI_2019_T_T_res']][4420:4430].sort_values(by='id'))
    #print('\nIndices computation check:\n', mimi[['id','ESTAT_2019_T_T_res','numerator_ESTAT_2019_T_T_res', 'ESTAT_flow_BMI_2019_T_T_res']][50:100].sort_values(by='id'))
    #print(mimi[mimi['numerator_UN_2019_T_T_res_count']==True][['id','UN_2019_T_T_res','numerator_UN_2019_T_T_res', 'UN_flow_BMI_2019_T_T_res']].sort_values(by='id', ascending=False))
    #print(mimi[mimi['numerator_UN_2019_T_T_res_count']==False][['id','UN_2019_T_T_res','numerator_UN_2019_T_T_res', 'UN_flow_BMI_2019_T_T_res']].sort_values(by='id', ascending=True)[22000:22050])
    #print(mimi[mimi.id==175][['id','UN_2019_T_T_res','numerator_UN_2019_T_T_res', 'UN_flow_BMI_2019_T_T_res']])

    mimi = num_vars(mimi)

    #print('Process successfully terminated.')

    return mimi, mimi_tot_indices


