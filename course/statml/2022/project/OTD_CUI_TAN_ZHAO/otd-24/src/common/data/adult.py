import os

import numpy as np
import torch.nn.functional as F
import torch
import numpy
import pandas
import pandas as pd
from torchvision.datasets.utils import download_url

from lib.src.data_utils import DATA_FOLDER
from lib.src.os_utils import safe_makedirs

ADULT_RAW_COL_NAMES = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
                       "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
                       "hours-per-week", "native-country", "income", ]
# this are categorical column indices; doesnot include sex and income column (these are binary)
ADULT_RAW_COL_FACTOR = [1, 3, 5, 6, 7, 8, 13]


def maybe_download():
    path = os.path.join(DATA_FOLDER, "adult")
    safe_makedirs(path)
    download_url("http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", path)
    download_url("http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test", path)

def load_adult_open(val_size=0.0):
    maybe_download()
    # taken from https://github.com/dcmoyer/inv-rep/blob/master/src/uci_data.py

    # load data as pandas table
    train_data = pandas.read_table(os.path.join(DATA_FOLDER, "adult", "adult.data"), delimiter=", ",
                                   header=None, names=ADULT_RAW_COL_NAMES, na_values="?",
                                   keep_default_na=False)
    test_data = pandas.read_table(os.path.join(DATA_FOLDER, "adult", "adult.test"), delimiter=", ",
                                  header=None, names=ADULT_RAW_COL_NAMES, na_values="?",
                                  keep_default_na=False, skiprows=1)

    # drop missing val
    train_data.dropna(inplace=True)
    test_data.dropna(inplace=True)

    # concatenate and binarize categorical variables
    all_data = pandas.concat([train_data, test_data])
    all_data = pandas.get_dummies(all_data,
                                  columns=[ADULT_RAW_COL_NAMES[i] for i in ADULT_RAW_COL_FACTOR])
    # fix binary variables now
    all_data.loc[all_data.income == ">50K", "income"] = 1
    all_data.loc[all_data.income == ">50K.", "income"] = 1
    all_data.loc[all_data.income == "<=50K", "income"] = 0
    all_data.loc[all_data.income == "<=50K.", "income"] = 0

    #all other are numerical beside sex?
    all_data.loc[all_data.sex == "Female", "sex"] = 0
    all_data.loc[all_data.sex == "Male", "sex"] = 1
    cutoff = train_data.shape[0]

    workclass=all_data["workclass_Private"]
    race=all_data[["race_Black","race_White","race_Other"]]
    sex=all_data["sex"]
    occupation=all_data["occupation_Sales"]
    education=all_data[["education_Preschool","education_Masters","education_Doctorate","education_Bachelors","education_9th","education_10th","education_11th","education_12th"]]
    #two=[[0,0],[0,1],[1,0],[1,1]]
    # sec_mix_occ=all_data[["sex","occupation_Sales"]]
    # sec_occ=np.zeros(len(sex))
    # for i in range(len(two)):
    #     sec_occ[sec_mix_occ[(sec_mix_occ["sex"]==two[i][0])&(sec_mix_occ["occupation_Sales"]==two[i][1])].index.tolist()]=i




    all_c={"sex":sex.to_numpy(),
           "race":race.to_numpy().argmax(axis=1),
           "occupation":occupation.to_numpy(),
           "education":education.to_numpy().argmax(axis=1),
           "workclass":workclass.to_numpy()}


    names=["sex","race","occupation","education","workclass"]
    meta=[2,3,2,8,2]
    mix_c=adult_mix(pd.DataFrame(all_c),meta,names)
    #remove the target column and sensitive attribute column

    mix_c.update(all_c)
    mix_c=pd.DataFrame(mix_c)
    all_c=pd.DataFrame(all_c)

    #mix_c=pd.merge([all_c, mix_c])
    #nan_row=mix_c[pd.isna(mix_c)]
    all_x = all_data.iloc[:, (all_data.columns != "income") & (all_data.columns != "sex")& (all_data.columns != "race_Black")& (all_data.columns != "race_White")& (all_data.columns != "race_Other")]

    all_labels = all_data.iloc[:, all_data.columns == "income"]

    # col_valid = [len(all_x.iloc[:, all_x.columns==x].unique()) > 1 for x in all_x.columns]
    # all_x = all_x.iloc[:, col_valid]

    # normalization
    maxes = all_x.max(axis=0)
    all_x = all_x / maxes

    train_data = all_x[:cutoff]
    train_c = all_c[:cutoff]
    train_mix_c = mix_c[:cutoff]
    train_labels = all_labels[:cutoff]

    test_data = all_x[cutoff:]
    test_c = all_c[cutoff:]
    test_mix_c = mix_c[cutoff:]
    test_labels = all_labels[cutoff:]

    # split off validation data (we keep val_size for training 0, so this code is never run)
    if val_size != 0:
        # # shuffle
        # train_data.sample(frac=1, random_state=0)
        # train_c.sample(frac=1, random_state=0)
        # train_labels.sample(frac=1, random_state=0)
        # split the train data to get some val
        val_cutoff = int((1 - val_size) * train_data.shape[0])

        val_data = train_data.iloc[val_cutoff:, :]
        train_data = train_data.iloc[:val_cutoff, :]

        val_labels = train_labels.iloc[val_cutoff:, :]
        train_labels = train_labels.iloc[:val_cutoff, :]

        val_c = train_c.iloc[val_cutoff:, :]
        val_mix_c = train_mix_c.iloc[val_cutoff:, :]

        train_c = train_c.iloc[:val_cutoff, :]
        train_mix_c = train_mix_c.iloc[:val_cutoff, :]

    return {"train": (
        train_data.to_numpy(),
        train_c.to_numpy(dtype=numpy.int),
        train_labels.to_numpy(dtype=numpy.int),
        train_mix_c.to_numpy(dtype=numpy.int),
    ), "valid": None if val_size == 0 else (
        val_data.to_numpy(),
        val_c.to_numpy(dtype=numpy.int),
        val_labels.to_numpy(dtype=numpy.int),
        val_mix_c.to_numpy(dtype=numpy.int),
    ), "test": (
        test_data.to_numpy(),
        test_c.to_numpy(dtype=numpy.int),
        test_labels.to_numpy(dtype=numpy.int),
        test_mix_c.to_numpy(dtype=numpy.int),
    )}
def adult_mix(c,meta,names):
    num_c=len(meta)
    #mix two
    c_out_all={}
    for i in range(num_c):
        for j in range(i+1,num_c):
            cat_i=np.arange(meta[i])
            cat_j=np.arange(meta[j])
            n_mix=meta[i]*meta[j]
            mix=[]
            for ii in cat_i:
                for jj in cat_j:
                    mix.append([ii,jj])
            c_out=np.zeros(len(c[names[0]]))
            for k in range(n_mix):
                c_out[c[(c[names[i]]==mix[k][0])&(c[names[j]]==mix[i][1])].index.tolist()]=k
            c_out_all[names[i]+'_'+names[j]]=c_out

    for i in range(num_c):
        for j in range(i+1,num_c):
            for a in range(j+1,num_c):
                cat_i=np.arange(meta[i])
                cat_j=np.arange(meta[j])
                cat_a=np.arange(meta[a])
                n_mix=meta[i]*meta[j]*meta[a]
                mix=[]
                for ii in cat_i:
                    for jj in cat_j:
                        for aa in cat_a:
                            mix.append([ii,jj,aa])
                c_out=np.zeros(len(c[names[0]]))
                for k in range(n_mix):
                    c_out[c[(c[names[i]]==mix[k][0])&(c[names[j]]==mix[i][1])&(c[names[a]]==mix[i][2])].index.tolist()]=k
                c_out_all[names[i]+'_'+names[j]+'_'+names[a]]=c_out
    for i in range(num_c):
        for j in range(i+1,num_c):
            for a in range(j+1,num_c):
                for b in range(a+1,num_c):
                    cat_i=np.arange(meta[i])
                    cat_j=np.arange(meta[j])
                    cat_a=np.arange(meta[a])
                    cat_b=np.arange(meta[b])

                    n_mix=meta[i]*meta[j]*meta[a]*meta[b]
                    mix=[]
                    for ii in cat_i:
                        for jj in cat_j:
                            for aa in cat_a:
                                for bb in cat_b:
                                    mix.append([ii,jj,aa,bb])
                    c_out=np.zeros(len(c[names[0]]))
                    for k in range(n_mix):
                        c_out[c[(c[names[i]]==mix[k][0])&(c[names[j]]==mix[i][1])&(c[names[a]]==mix[i][2])&(c[names[b]]==mix[i][3])].index.tolist()]=k
                    c_out_all[names[i]+'_'+names[j]+'_'+names[a]+'_'+names[b]]=c_out
    for i in range(num_c):
        for j in range(i+1,num_c):
            for a in range(j+1,num_c):
                for b in range(a+1,num_c):
                    for d in range(b+1,num_c):
                        cat_i=np.arange(meta[i])
                        cat_j=np.arange(meta[j])
                        cat_a=np.arange(meta[a])
                        cat_b=np.arange(meta[b])
                        cat_d=np.arange(meta[d])

                        n_mix=meta[i]*meta[j]*meta[a]*meta[b]*meta[d]
                        mix=[]
                        for ii in cat_i:
                            for jj in cat_j:
                                for aa in cat_a:
                                    for bb in cat_b:
                                        for dd in cat_d:
                                            mix.append([ii,jj,aa,bb,dd])
                        c_out=np.zeros(len(c[names[0]]))
                        for k in range(n_mix):
                            c_out[c[(c[names[i]]==mix[k][0])&(c[names[j]]==mix[i][1])&(c[names[a]]==mix[i][2])&(c[names[b]]==mix[i][3])&(c[names[d]]==mix[i][4])].index.tolist()]=k
                        c_out_all[names[i]+'_'+names[j]+'_'+names[a]+'_'+names[b]+'_'+names[b]]=c_out

    #return pd.DataFrame(c_out_all)
    return c_out_all

def load_adult_open_perfect(val_size=0.0):
    maybe_download()
    # taken from https://github.com/dcmoyer/inv-rep/blob/master/src/uci_data.py

    # load data as pandas table
    train_data = pandas.read_table(os.path.join(DATA_FOLDER, "adult", "adult.data"), delimiter=", ",
                                   header=None, names=ADULT_RAW_COL_NAMES, na_values="?",
                                   keep_default_na=False)
    test_data = pandas.read_table(os.path.join(DATA_FOLDER, "adult", "adult.test"), delimiter=", ",
                                  header=None, names=ADULT_RAW_COL_NAMES, na_values="?",
                                  keep_default_na=False, skiprows=1)

    # drop missing val
    train_data.dropna(inplace=True)
    test_data.dropna(inplace=True)

    # concatenate and binarize categorical variables
    all_data = pandas.concat([train_data, test_data])
    all_data = pandas.get_dummies(all_data,
                                  columns=[ADULT_RAW_COL_NAMES[i] for i in ADULT_RAW_COL_FACTOR])
    # fix binary variables now
    all_data.loc[all_data.income == ">50K", "income"] = 1
    all_data.loc[all_data.income == ">50K.", "income"] = 1
    all_data.loc[all_data.income == "<=50K", "income"] = 0
    all_data.loc[all_data.income == "<=50K.", "income"] = 0

    #all other are numerical beside sex?
    all_data.loc[all_data.sex == "Female", "sex"] = 0
    all_data.loc[all_data.sex == "Male", "sex"] = 1
    cutoff = train_data.shape[0]
    workclass=all_data["workclass_Private"]

    race=all_data[["race_Black","race_White","race_Other"]]
    sex=all_data["sex"]
    occupation=all_data["occupation_Sales"]
    education=all_data[["education_Preschool","education_Masters","education_Doctorate","education_Bachelors","education_9th","education_10th","education_11th","education_12th"]]

    all_c={"sex":sex.to_numpy(),
           "race":race.to_numpy().argmax(axis=1),
           "occupation":occupation.to_numpy(),
           "education":education.to_numpy().argmax(axis=1),
           "workclass":workclass.to_numpy()}
    all_c=pd.DataFrame(all_c)
    #remove the target column and sensitive attribute column
    all_x = all_data.iloc[:, (all_data.columns != "income") & (all_data.columns != "sex")& (all_data.columns != "race_Black")& (all_data.columns != "race_White")& (all_data.columns != "race_Other")]

    all_labels = all_data.iloc[:, all_data.columns == "income"]

    # col_valid = [len(all_x.iloc[:, all_x.columns==x].unique()) > 1 for x in all_x.columns]
    # all_x = all_x.iloc[:, col_valid]

    # normalization
    maxes = all_x.max(axis=0)
    all_x = all_x / maxes

    train_data = all_x[:cutoff]
    train_c = all_c[:cutoff]
    train_labels = all_labels[:cutoff]

    test_data = all_x[cutoff:]
    test_c = all_c[cutoff:]
    test_labels = all_labels[cutoff:]

    # split off validation data (we keep val_size for training 0, so this code is never run)
    if val_size != 0:
        # # shuffle
        # train_data.sample(frac=1, random_state=0)
        # train_c.sample(frac=1, random_state=0)
        # train_labels.sample(frac=1, random_state=0)
        # split the train data to get some val
        val_cutoff = int((1 - val_size) * train_data.shape[0])

        val_data = train_data.iloc[val_cutoff:, :]
        train_data = train_data.iloc[:val_cutoff, :]

        val_labels = train_labels.iloc[val_cutoff:, :]
        train_labels = train_labels.iloc[:val_cutoff, :]

        val_c = train_c.iloc[val_cutoff:, :]
        train_c = train_c.iloc[:val_cutoff, :]

    return {"train": (
        train_data.to_numpy(),
        train_c.to_numpy(dtype=numpy.int),
        train_labels.to_numpy(dtype=numpy.int),
    ), "valid": None if val_size == 0 else (
        val_data.to_numpy(),
        val_c.to_numpy(dtype=numpy.int),
        val_labels.to_numpy(dtype=numpy.int),
    ), "test": (
        test_data.to_numpy(),
        test_c.to_numpy(dtype=numpy.int),
        test_labels.to_numpy(dtype=numpy.int),
    )}
def load_adult(val_size=0.0):
    maybe_download()
    # taken from https://github.com/dcmoyer/inv-rep/blob/master/src/uci_data.py

    # load data as pandas table
    train_data = pandas.read_table(os.path.join(DATA_FOLDER, "adult", "adult.data"), delimiter=", ",
                                   header=None, names=ADULT_RAW_COL_NAMES, na_values="?",
                                   keep_default_na=False)
    test_data = pandas.read_table(os.path.join(DATA_FOLDER, "adult", "adult.test"), delimiter=", ",
                                  header=None, names=ADULT_RAW_COL_NAMES, na_values="?",
                                  keep_default_na=False, skiprows=1)

    # drop missing val
    train_data.dropna(inplace=True)
    test_data.dropna(inplace=True)

    # concatenate and binarize categorical variables
    all_data = pandas.concat([train_data, test_data])
    all_data = pandas.get_dummies(all_data,
                                  columns=[ADULT_RAW_COL_NAMES[i] for i in ADULT_RAW_COL_FACTOR])
    # fix binary variables now
    all_data.loc[all_data.income == ">50K", "income"] = 1
    all_data.loc[all_data.income == ">50K.", "income"] = 1
    all_data.loc[all_data.income == "<=50K", "income"] = 0
    all_data.loc[all_data.income == "<=50K.", "income"] = 0

    all_data.loc[all_data.sex == "Female", "sex"] = 0
    all_data.loc[all_data.sex == "Male", "sex"] = 1
    cutoff = train_data.shape[0]

    all_x = all_data.iloc[:, (all_data.columns != "income") & (all_data.columns != "sex")]
    all_c = all_data.iloc[:, all_data.columns == "sex"]
    all_labels = all_data.iloc[:, all_data.columns == "income"]

    # col_valid = [len(all_x.iloc[:, all_x.columns==x].unique()) > 1 for x in all_x.columns]
    # all_x = all_x.iloc[:, col_valid]

    # normalization
    maxes = all_x.max(axis=0)
    all_x = all_x / maxes

    train_data = all_x[:cutoff]
    train_c = all_c[:cutoff]
    train_labels = all_labels[:cutoff]

    test_data = all_x[cutoff:]
    test_c = all_c[cutoff:]
    test_labels = all_labels[cutoff:]

    # split off validation data (we keep val_size for training 0, so this code is never run)
    if val_size != 0:
        # # shuffle
        # train_data.sample(frac=1, random_state=0)
        # train_c.sample(frac=1, random_state=0)
        # train_labels.sample(frac=1, random_state=0)

        val_cutoff = int((1 - val_size) * train_data.shape[0])

        val_data = train_data.iloc[val_cutoff:, :]
        train_data = train_data.iloc[:val_cutoff, :]

        val_labels = train_labels.iloc[val_cutoff:, :]
        train_labels = train_labels.iloc[:val_cutoff, :]

        val_c = train_c.iloc[val_cutoff:, :]
        train_c = train_c.iloc[:val_cutoff, :]

    return {"train": (
        train_data.to_numpy(),
        train_c.to_numpy(dtype=numpy.int),
        train_labels.to_numpy(dtype=numpy.int),
    ), "valid": None if val_size == 0 else (
        val_data.to_numpy(),
        val_c.to_numpy(dtype=numpy.int),
        val_labels.to_numpy(dtype=numpy.int),
    ), "test": (
        test_data.to_numpy(),
        test_c.to_numpy(dtype=numpy.int),
        test_labels.to_numpy(dtype=numpy.int),
    )}


if __name__ == "__main__":
    data = load_adult(0.2)
    print("Adult dataset:")
    print(f"Train size: {data['train'][0].shape}")
    if data['valid'] is not None:
        print(f"Val size: {data['valid'][0].shape}")
    print(f"Test size: {data['test'][0].shape}")
