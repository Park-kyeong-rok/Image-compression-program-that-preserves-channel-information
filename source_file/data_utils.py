import numpy as np
from sklearn.model_selection import train_test_split

def data_split(x, y, seed=40 ):
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=1 / 5, shuffle=True,stratify = y, random_state=40)
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=1/4, shuffle=True,stratify = train_y,  random_state = seed)
    return train_x, val_x, test_x, train_y, test_y, val_y

def load_data(data_name):
    train_x = np.load(f'processed_data/{data_name}_train_x.npy')
    train_y = np.load(f'processed_data/{data_name}_train_y.npy')

    # 값들의 범위가 0~1이 되도록 Normalization
    max = np.max(train_x)
    min = np.min(train_x)
    train_x = (train_x-min)/(max-min)

    val_x = (np.load(f'processed_data/{data_name}_val_x.npy')-min)/(max-min)
    val_y = np.load(f'processed_data/{data_name}_val_y.npy')

    test_x = (np.load(f'processed_data/{data_name}_test_x.npy')-min)/(max-min)
    test_y = np.load(f'processed_data/{data_name}_test_y.npy')


    return train_x, train_y, val_x, val_y, test_x, test_y