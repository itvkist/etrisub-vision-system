import os
import pandas as pd

def load_class_names(path="./vehicles/class_names_new.csv"):
    cn = pd.read_csv(path, header=None).values.reshape(-1)
    cn = cn.tolist()
    return cn

def separate_class(class_names):
    arr = []
    for idx, name in enumerate(class_names):
        splits = name.split(' ')
        make = splits[0]
        model = ' '.join(splits[1:-1])
        model_type = splits[-2]

        if model == 'General Hummer SUV':
            make = 'AM General'
            model = 'Hummer SUV'

        if model == 'Integra Type R':
            model_type = 'Type-R'

        if model_type == 'Z06' or model_type == 'ZR1':
            model_type = 'Convertible'

        if 'SRT' in model_type:
            model_type = 'SRT'

        if model_type == 'IPL':
            model_type = 'Coupe'

        year = splits[-1]
        arr.append((idx, make, model, model_type, year))

    arr = pd.DataFrame(arr, columns=['target', 'make', 'model', 'model_type', 'year'])
    return arr