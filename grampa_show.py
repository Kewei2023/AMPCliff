import pandas as pd
import numpy as np
import ipdb

if __name__=='__main__':
    
    bacteria_list=['E. faecium', 'S. aureus','K. pneumoniae','A. baumannii','P. aeruginosa','Enterobacter spp']
    
    data_ = pd.read_csv('./data/grampa.csv',index_col=0)
    data = data_[(data_.datasource_has_modifications) & (~data_.sequence.str.contains('C'))]
    # data = data[ ~(data.modifications.str.contains('disulfide'))]
    # ipdb.set_trace()
    print(f'removing sequences without modification infomation and C:{data.shape[0]}/{data_.shape[0]}')
    
    result = data.groupby(['bacterium', 'sequence']).agg({'value': lambda x: np.log10(np.mean(10 ** x))}).reset_index()
    print(f'remove duplicate sequences:{result.shape[0]}/{data.shape[0]}')
    
    filtered_result = result[result['bacterium'].isin(bacteria_list)]
    
    result['bacterium'].value_counts().to_csv('./data/grampa_v1_count.csv')
    
    counts = filtered_result['bacterium'].value_counts()
    counts.to_csv('./data/grampa_filtered_v1_count.csv')
    
    result_ = data_.groupby(['bacterium', 'sequence']).agg({'value': lambda x: np.log10(np.mean(10 ** x))}).reset_index()
    

    # ipdb.set_trace()
    # print(f'remove duplicate sequences:{result_.shape[0]}/{data_.shape[0]}')
    # result_['bacterium'].value_counts().to_csv('./regression/grampa_count.csv')
    exit()
    
    result['length'] = result['sequence'].apply(len)
    result.rename(columns={'sequence':'Sequence'},inplace=True) 
    result['Activity'] =6 - result['value']
    result.to_csv('./regression/grampa_v1.csv')




    df = result.dropna()
    e_coli = df.loc[df.bacterium.str.contains('E. coli')]
    s_aureus = df.loc[df.bacterium.str.contains('S. aureus')]

    # ipdb.set_trace()
    e_coli.to_csv('./regression/grampa_e_coli.csv')
    s_aureus.to_csv('./regression/grampa_s_aureus.csv')
    
    print(f'there are {e_coli.shape[0]} sequences in e.coli')
    print(f'there are {s_aureus.shape[0]} sequences in s_aureus')
    
    e_coli_7_25 = e_coli[(e_coli.length >= 7) & (e_coli.length <= 25)]
    e_coli_7_25['Idx'] = [i for i in range(e_coli_7_25.shape[0])]
    e_coli_7_25['ID'] = [i for i in range(e_coli_7_25.shape[0])]
    e_coli_7_25.to_csv('./regression/grampa_e_coli_7_25.csv',index=False)

    # exit()
    s_aureus_7_25 = s_aureus[(s_aureus.length >= 7) & (s_aureus.length <= 25)]
    s_aureus_7_25['Idx'] = [i for i in range(s_aureus_7_25.shape[0])]
    s_aureus_7_25['ID'] = [i for i in range(s_aureus_7_25.shape[0])]
    s_aureus_7_25.to_csv('./regression/grampa_s_aureus_7_25.csv',index=False)


    print(f'there are {e_coli_7_25.shape[0]}/{e_coli.shape[0]} sequences in e.coli are 7-25 length')
    print(f'there are {s_aureus_7_25.shape[0]}/{s_aureus.shape[0]} sequences in s_aureus are 7-25 length')
    


