import pandas as pd
import numpy as np
import pickle
import os

def reset_dict(dict_dir):
    open(dict_dir, 'wb').close()

def get_df_news(data_folder, news_to_read, format_cols):
    df = pd.read_parquet(data_folder + 'financial_data' + news_to_read + '.parquet.gzip')
    df = df.rename(columns = {'Article':'Text', 'Journalists':'Author'})
    return df[format_cols]

def get_dict(dict_dir):
    f = open(dict_dir,"r+b")
    if (os.stat(dict_dir).st_size == 0):
        data_dict = {}
    else :
        data_dict = pickle.load(f)
    f.close()
    return data_dict

def dump_dict(dct, dict_dir):
    f = open(dict_dir,"r+b")
    pickle.dump(dct, f)
    f.close()

def check_df_format(source_df, cols):
    assert(len(source_df.columns == len(cols))), 'Number of columns mismatch (needs 2 : Date and Article)'
    for i in range(len(cols)):
        assert(cols[i] == source_df.columns.values[i]), f"Column mismatch : {source_df.columns.values[i]} when {cols(i)} was expected"

        
def make_dict(source_df, companies_to_get, dict_dir, format_cols):
    data_dict = get_dict(dict_dir)
    keys = data_dict.keys()

    check_df_format(source_df, format_cols)
    companies_to_get = [company_name.lower() for company_name in companies_to_get]
    for company_name in companies_to_get:
        company_df = source_df[source_df[format_cols[0]].apply(lambda text : company_name in text.lower())]
        if company_name in keys:
            data_dict[company_name] = pd.merge(data_dict[company_name], company_df,how='outer', on=format_cols[0])
        else :
            data_dict[company_name] = company_df
    dump_dict(data_dict, dict_dir)
    
