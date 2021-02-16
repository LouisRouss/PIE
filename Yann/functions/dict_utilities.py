import pandas as pd
import numpy as np
import pickle
import os
from shutil import copyfile


def reset_dict(dict_dir):
    '''Empties the data dictionary situated in dict_dir directory'''
    open(dict_dir, 'wb').close()

def backup_dict(dict_dir, backup_dir):
    copyfile(dict_dir, backup_dir)
    
def get_dict(dict_dir):
    '''Gets dictionary from pkl file in dict_dir directory'''
    f = open(dict_dir,"r+b")
    if (os.stat(dict_dir).st_size == 0):
        data_dict = {}
    else :
        data_dict = pickle.load(f)
    f.close()
    return data_dict

def dump_dict(dct, dict_dir):
    '''Zips dict dictionary into a pkl file in dict_dir directory'''
    f = open(dict_dir,"r+b")
    pickle.dump(dct, f)
    f.close()


def add_to_dict(df_to_add, search_word, dict_dir):
    '''Adds df_to_add dataframe to data dictionary in dict_dir with ticker search_word
    format_cols : temporary argument until we agree on the format of dataframe'''
    data_dict = get_dict(dict_dir)
    keys = data_dict.keys()
    
    search_word = search_word.lower()
    if search_word in keys:
        data_dict[search_word] = pd.concat([data_dict[search_word],df_to_add]).drop_duplicates()
    else :
        data_dict[search_word] = df_to_add
    dump_dict(data_dict, dict_dir)
    
