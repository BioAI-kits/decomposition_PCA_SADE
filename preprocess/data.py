import pandas as pd 
import numpy as np 
import os

def split_data_to_matrix(file_path, dimen_len=500, save_file="./result/matrix.txt"):
    """
    file_path: raw data file path
    dimen_len: dimension length, default set 500
    save_file: save matrix data
    """
    df = pd.read_csv(file_path)
    raw_data = df.iloc[:, 0]
    matrix_data = np.resize(raw_data, (df.shape[0]//dimen_len, dimen_len))
    
    try:
       os.makedirs("result") 
    except:
        pass
    
    np.savetxt("./data/matrix.txt", matrix_data)
    
    return matrix_data

