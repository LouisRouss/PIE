import pandas as pd
import numpy as np
import pickle
import os

def reset_dict(dict_dir):
    open(dict_dir, 'wb').close()

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

def check_df_format(df_to_add, cols):
    assert(len(df_to_add.columns == len(cols))), 'Number of columns mismatch (needs 2 : Date and Article)'
    for i in range(len(cols)):
        assert(cols[i] == df_to_add.columns.values[i]), f"Column mismatch : {df_to_add.columns.values[i]} when {cols(i)} was expected"


def add_to_dict(df_to_add, search_word, dict_dir, format_cols):
    data_dict = get_dict(dict_dir)
    keys = data_dict.keys()
    
    check_df_format(df_to_add, format_cols)
    search_word = search_word.lower()
    if search_word in keys:
        data_dict[search_word] = pd.merge(data_dict[search_word],df_to_add ,how='outer', on=format_cols[0])
    else :
        data_dict[search_word] = df_to_add
    dump_dict(data_dict, dict_dir)
    
