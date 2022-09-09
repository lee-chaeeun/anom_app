import plotly.graph_objects as go
import plotly.io as pio

import flask
from flask import Flask,render_template,request, flash, redirect
import plotly
import plotly.express as px 

import pandas as pd
import numpy as np

import csv
import codecs
import subprocess
import sys
import os
import json
import pickle
import glob
import pathlib
import logging
import datetime, operator, time
import timeit
import math
import ast

from src.datasets import Dataset
from src.datasets.entities_names import smap_entities
from src.datasets.entities_names import msl_entities
from src.datasets.dataset import get_events

#directory = os.path.join("FA","mvts-ano-eval")
directory =os.getcwd()


"""
@author of dataset Classes: Astha Garg 10/19 - edited to extract test data for plotting
@author of plotting functions: Chae Eun Lee 09/07
"""

class Skab(Dataset):

    def __init__(self, seed: int, entity=None, verbose=False):
        """
        :param seed: for repeatability
        :param entity: for compatibility with multi-entity datasets. Value provided will be ignored. self.entity will be
         set to same as dataset name for single entity datasets
        """
        name = "skab"
        super().__init__(name=name, file_name="anomaly-free.csv")
        #EDIT PATH to fit folder structure 
        root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                           directory,"data", "raw", "skab")
        self.raw_path_train = os.path.join(root, "anomaly-free", "anomaly-free.csv")
        self.raw_paths_test = [os.path.join(root, folder) for folder in ["other", "valve1", "valve2"]]
        self.seed = seed
        self.verbose = verbose
        self.causes = None


    def load(self):
        # 1 is the outlier, all other digits are normal
        OUTLIER_CLASS = 1
        train_df: pd.DataFrame = pd.read_csv(self.raw_path_train, index_col='datetime',sep=';',parse_dates=True)
        print("train df len {}".format(len(train_df)))

        test_df = []
        orig_test_len = 0
        for path_name in self.raw_paths_test:
            path_files = os.listdir(path_name)
            for file_name in path_files:
                file_path = os.path.join(path_name, file_name)
                df = pd.read_csv(file_path, index_col='datetime',sep=';', parse_dates=True)
                # print("df of length {} clipped to {}".format(len(df), 100*(len(df)//100)))
                orig_test_len += len(df)
                df = df.head(100*(len(df)//100))
                test_df.append(df)
        test_df = pd.concat(test_df, ignore_index=True, axis=0)
        print("orig test len {}, modified len {}".format(orig_test_len, len(test_df)	))        
        self._data = test_df

class Smap_entity(Dataset):

    def __init__(self, seed: int, entity="A-1", remove_unique=False, verbose=False):
        """
        :param seed: for repeatability
        """
        if entity in smap_entities:
            name = "smap-" + entity
        else:
            name = "not_found"
            print("Entity name not recognized")
        super().__init__(name=name, file_name="A-1.npy")
        #EDIT PATH to fit folder structure 
        self.base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                      directory,"data", "raw", "smap_msl")
        self.seed = seed
        self.remove_unique = remove_unique
        self.entity = entity
        self.verbose = verbose

    @staticmethod
    def one_hot_encoding(df, col_name):
        with_dummies = pd.get_dummies(df[col_name])
        with_dummies = with_dummies.rename(columns={name: col_name + "_" + str(name) for name in with_dummies.columns})
        new_df = pd.concat([df, with_dummies], axis=1).drop(col_name, axis=1)
        return new_df

    def load(self):
        # 1 is the outlier, all other digits are normal
        OUTLIER_CLASS = 1
        train_values = np.load(os.path.join(self.base_path, "train", self.entity + ".npy"))
        train_df = pd.DataFrame(train_values)
        test_values = np.load(os.path.join(self.base_path, "test", self.entity + ".npy"))
        test_df = pd.DataFrame(test_values)
        train_df["y"] = np.zeros(train_df.shape[0])

        # Get test anomaly labels
        test_labels = np.zeros(test_df.shape[0])
        labels_df = pd.read_csv(os.path.join(self.base_path, "labeled_anomalies.csv"), header=0, sep=",")
        entity_attacks = labels_df[labels_df["chan_id"] == self.entity]["anomaly_sequences"].values
        entity_attacks = ast.literal_eval(entity_attacks[0])
        for sequence in entity_attacks:
            test_labels[sequence[0]:(sequence[1] + 1)] = 1
        test_df["anomalies"] = test_labels    
        self._data = test_df

class Msl_entity(Dataset):

    def __init__(self, seed: int, entity="C-1", remove_unique=False, verbose=False):
        """
        :param seed: for repeatability
        """
        if entity in msl_entities:
            name = "msl-" + entity
        else:
            name = "not_found"
            print("Entity name not recognized")
        super().__init__(name=name, file_name="C-1.npy")
        self.base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                      directory,"data", "raw", "smap_msl")
        self.seed = seed
        self.remove_unique = remove_unique
        self.entity = entity
        self.verbose = verbose

    @staticmethod
    def one_hot_encoding(df, col_name):
        with_dummies = pd.get_dummies(df[col_name])
        with_dummies = with_dummies.rename(columns={name: col_name + "_" + str(name) for name in with_dummies.columns})
        new_df = pd.concat([df, with_dummies], axis=1).drop(col_name, axis=1)
        return new_df

    def load(self):
        # 1 is the outlier, all other digits are normal
        OUTLIER_CLASS = 1
        train_values = np.load(os.path.join(self.base_path, "train", self.entity + ".npy"))
        train_df = pd.DataFrame(train_values)
        test_values = np.load(os.path.join(self.base_path, "test", self.entity + ".npy"))
        test_df = pd.DataFrame(test_values)
        train_df["y"] = np.zeros(train_df.shape[0])

        # Get test anomaly labels
        test_labels = np.zeros(test_df.shape[0])
        labels_df = pd.read_csv(os.path.join(self.base_path, "labeled_anomalies.csv"), header=0, sep=",")
        entity_attacks = labels_df[labels_df["chan_id"] == self.entity]["anomaly_sequences"].values
        entity_attacks = ast.literal_eval(entity_attacks[0])
        for sequence in entity_attacks:
            test_labels[sequence[0]:(sequence[1] + 1)] = 1
        test_df["anomalies"] = test_labels
        self._data = test_df    
        
        
class Smd_entity(Dataset):

    def __init__(self, seed: int, entity="machine-1-1", remove_unique=False, verbose=False):
        """
        :param seed: for repeatability
        """
        name = "smd-" + entity
        super().__init__(name=name, file_name="machine-1-1.txt")
        self.base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                      directory,"data", "raw", "ServerMachineDataset")
        self.seed = seed
        self.remove_unique = remove_unique
        self.entity = entity
        self.verbose = verbose

    @staticmethod
    def one_hot_encoding(df, col_name):
        with_dummies = pd.get_dummies(df[col_name])
        with_dummies = with_dummies.rename(columns={name: col_name + "_" + str(name) for name in with_dummies.columns})
        new_df = pd.concat([df, with_dummies], axis=1).drop(col_name, axis=1)
        return new_df

    def load(self):
        # 1 is the outlier, all other digits are normal
        OUTLIER_CLASS = 1
        train_df = pd.read_csv(os.path.join(self.base_path, "train", self.entity + ".txt"), header=None, sep=",",
                               dtype=np.float32)
        test_df = pd.read_csv(os.path.join(self.base_path, "test", self.entity + ".txt"), header=None, sep=",",
                              dtype=np.float32)
        train_df["y"] = np.zeros(train_df.shape[0])

        # Get test anomaly labels
        test_labels = np.genfromtxt(os.path.join(self.base_path, "test_label", self.entity + ".txt"), dtype=np.float32,
                                    delimiter=',')
        test_df["anomalies"] = test_labels
        self._data = test_df  


        
class Swat(Dataset):

    def __init__(self, seed: int, shorten_long=True, remove_unique=False, entity=None, verbose=False, one_hot=False):
        """
        :param seed: for repeatability
        :param entity: for compatibility with multi-entity datasets. Value provided will be ignored. self.entity will be
         set to same as dataset name for single entity datasets
        """
        if shorten_long:
            name = "swat"
        else:
            name = "swat-long"
        super().__init__(name=name, file_name="SWaT_Dataset_Normal_v1.csv")
        root = os.path.join(os.getcwd(),"data", "raw", "swat", "raw")
        self.raw_path_train = os.path.join(root, "SWaT_Dataset_Normal_v1.csv")
        self.raw_path_test = os.path.join(root, "SWaT_Dataset_Attack_v0.csv")

        if not os.path.isfile(self.raw_path_train):
            df = pd.read_excel(os.path.join(root, "SWaT_Dataset_Normal_v1.xlsx"))
            df.to_csv(self.raw_path_train, index=False)
            
        if not os.path.isfile(self.raw_path_test):
            df = pd.read_excel(os.path.join(root, "SWaT_Dataset_Attack_v0.xlsx"))
            df.to_csv(self.raw_path_test, index=False)

        self.seed = seed
        self.shorten_long = shorten_long
        self.remove_unique = remove_unique
        self.verbose = verbose
        self.one_hot = one_hot

    def load(self):
        # 1 is the outlier, all other digits are normal
        OUTLIER_CLASS = 1
        
        #skip first row because it is empty
        test_df: pd.DataFrame = pd.read_csv(self.raw_path_test, skiprows = 1)
        train_df: pd.DataFrame = pd.read_csv(self.raw_path_train, skiprows = 1)

        train_df = train_df.rename(columns={col: col.strip() for col in train_df.columns})
        test_df = test_df.rename(columns={col: col.strip() for col in test_df.columns})

        train_df["y"] = train_df["Normal/Attack"].replace(to_replace=["Normal", "Attack", "A ttack"], value=[0, 1, 1])
        train_df = train_df.drop(columns=["Normal/Attack", "Timestamp"], axis=1)
        test_df["y"] = test_df["Normal/Attack"].replace(to_replace=["Normal", "Attack", "A ttack"], value=[0, 1, 1])
        test_df = test_df.drop(columns=["Normal/Attack", "Timestamp"], axis=1)

        # one-hot-encoding stuff
        if self.one_hot:
            keywords = {col_name: "".join([s for s in col_name if not s.isdigit()]) for col_name in train_df.columns}
            cat_cols = [col for col in keywords.keys() if keywords[col] in ["P", "MV", "UV"]]
            one_hot_cols = [col for col in cat_cols if train_df[col].nunique() >= 3 or test_df[col].nunique() >= 3]
            print(one_hot_cols)
            one_hot_encoded = Dataset.one_hot_encoding(pd.concat([train_df, test_df], axis=0, join="inner"),
                                                       col_names=one_hot_cols)
            train_df = one_hot_encoded.iloc[:len(train_df)]
            test_df = one_hot_encoded.iloc[len(train_df):]

        # shorten the extra long anomaly to 550 points
        if self.shorten_long:
            long_anom_start = 227828
            long_anom_end = 263727
            test_df = test_df.drop(test_df.loc[(long_anom_start + 551):(long_anom_end + 1)].index,
                                   axis=0).reset_index(drop=True)
        causes_channels_names = [["MV101"], ["P102"], ["LIT101"], [], ["AIT202"], ["LIT301"], ["DPIT301"],
                                 ["FIT401"], [], ["MV304"], ["MV303"], ["LIT301"], ["MV303"], ["AIT504"],
                                 ["AIT504"], ["MV101", "LIT101"], ["UV401", "AIT502", "P501"], ["P602", "DPIT301",
                                                                                                "MV302"],
                                 ["P203", "P205"], ["LIT401", "P401"], ["P101", "LIT301"], ["P302", "LIT401"],
                                 ["P201", "P203", "P205"], ["LIT101", "P101", "MV201"], ["LIT401"], ["LIT301"],
                                 ["LIT101"], ["P101"], ["P101", "P102"], ["LIT101"], ["P501", "FIT502"],
                                 ["AIT402", "AIT502"], ["FIT401", "AIT502"], ["FIT401"], ["LIT301"]]
                                 
        self._data = test_df  


class Wadi(Dataset):

    def __init__(self, seed: int, remove_unique=False, entity=None, verbose=False, one_hot=False):
        """
        :param seed: for repeatability
        :param entity: for compatibility with multi-entity datasets. Value provided will be ignored. self.entity will be
         set to same as dataset name for single entity datasets
        """
        super().__init__(name="wadi", file_name="WADI_14days.csv")
        self.raw_path_train = os.path.join(os.getcwd(),"data", "raw", "wadi", "raw", "WADI_14days.csv")
        self.raw_path_test = os.path.join(os.getcwd(),"data", "raw", "wadi", "raw", "WADI_attackdata.csv")
        self.anomalies_path = os.path.join(os.getcwd(),"data", "raw", "wadi", "raw", "WADI_anomalies.csv")
        self.seed = seed
        self.remove_unique = remove_unique
        self.verbose = verbose
        self.one_hot = one_hot

    def load(self):
        # 1 is the outlier, all other digits are normal
        OUTLIER_CLASS = 1
        test_df: pd.DataFrame = pd.read_csv(self.raw_path_test, header=0)
        train_df: pd.DataFrame = pd.read_csv(self.raw_path_train, header=3)

        # Removing 4 columns who only contain nans (data missing from the csv file)
        nan_columns = [r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_LS_001_AL',
                       r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_LS_002_AL',
                       r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_P_001_STATUS',
                       r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_P_002_STATUS']
        train_df = train_df.drop(nan_columns, axis=1)
        test_df = test_df.drop(nan_columns, axis=1)

        train_df = train_df.rename(columns={col: col.split('\\')[-1] for col in train_df.columns})
        test_df = test_df.rename(columns={col: col.split('\\')[-1] for col in test_df.columns})

        # Adding anomaly labels as a column in the dataframes
        ano_df = pd.read_csv(self.anomalies_path, header=0)
        #train_df["y"] = np.zeros(train_df.shape[0])
        test_df["anomalies"] = np.zeros(test_df.shape[0])
        causes = []
        
        for i in range(ano_df.shape[0]):
            ano = ano_df.iloc[i, :][["Start_time", "End_time", "Date"]]
            start_row = np.where((test_df["Time"].values == ano["Start_time"]) &
                                 (test_df["Date"].values == ano["Date"]))[0][0]
            end_row = np.where((test_df["Time"].values == ano["End_time"]) &
                               (test_df["Date"].values == ano["Date"]))[0][0]
            test_df["anomalies"].iloc[start_row:(end_row + 1)] = np.ones(1 + end_row - start_row)
            #causes.append(ano_df.iloc[i, :]["Causes"])

        test_df = test_df.iloc[:,1:] #drop first two columns indicating row.
        test_df["Date"] = test_df['Date'].astype(str) +" "+ test_df["Time"]
        test_df = test_df.drop('Time', axis=1)
        
        self._data = test_df

class Damadics(Dataset):

    def __init__(self, seed: int, remove_unique=False, entity=None, verbose=False, drop_init_test=False):
        """
        :param seed: for repeatability
        :param entity: for compatibility with multi-entity datasets. Value provided will be ignored. self.entity will be
         set to same as dataset name for single entity datasets
        """
        if drop_init_test:
            name = "damadics-s"
        else:
            name = "damadics"
        super().__init__(name=name, file_name="31102001.txt")
        train_filenames = ["31102001.txt"] + ["0"+str(i) + "112001.txt" for i in range(1, 9)]
        test_filenames = ["09112001.txt", "17112001.txt", "20112001.txt"]
        self.test_dates = [date[:-4] for date in test_filenames] # will be used to add labels to dataframes
        self.raw_paths_train = [os.path.join(os.getcwd(), "data", "raw", "damadics", "raw", filename) for filename in train_filenames]
        self.raw_paths_test = [os.path.join(os.getcwd(),
                                           "data", "raw", "damadics", "raw", filename) for filename in test_filenames]
        self.anomalies_path = os.path.join(os.getcwd(), "data", "raw", "damadics", "raw", "DAMADICS_anomalies.csv")
        self.seed = seed
        self.remove_unique = remove_unique
        self.verbose = verbose
        self.drop_init_test = drop_init_test

    @staticmethod
    def one_hot_encoding(df, col_name):
        with_dummies = pd.get_dummies(df[col_name])
        with_dummies = with_dummies.rename(columns={name: col_name + "_" + str(name) for name in with_dummies.columns})
        new_df = pd.concat([df, with_dummies], axis=1).drop(col_name, axis=1)
        return new_df

    def load(self):
        # 1 is the outlier, all other digits are normal
        OUTLIER_CLASS = 1
        train_dataframes = [pd.read_csv(path, header=None, sep="\t") for path in self.raw_paths_train]
        test_dataframes = [pd.read_csv(path, header=None, sep="\t") for path in self.raw_paths_test]

        # Adding anomaly labels as a column in the dataframes
        ano_df = pd.read_csv(self.anomalies_path, header=0, dtype=str)
        for df in test_dataframes:
            df["y"] = np.zeros(df.shape[0])
        for i in range(ano_df.shape[0]):
            ano = ano_df.iloc[i, :][["Start_time", "End_time", "Date"]]
            date = ano["Date"]
            df_idx = self.test_dates.index(date)
            start_row = int(ano["Start_time"])
            end_row = int(ano["End_time"])
            test_dataframes[df_idx]["y"].iloc[start_row:(end_row + 1)] = np.ones(1 + end_row - start_row)
            
        #train_df: pd.DataFrame = pd.concat(train_dataframes, axis=0, ignore_index=True)
        test_df: pd.DataFrame = pd.concat(test_dataframes, axis=0, ignore_index=True)
        
        self._data = test_df      


def plot_skab():
    
    seed = 0
    ds = Skab(seed=seed)
    test_df = ds.data()
  
    title = 'Original Time Series Plot of Skab'    
    fig = px.line(test_df, x = test_df.index, y = list(test_df.columns), width=1000, height=500, title=title)
    #fig_json = fig.to_json()
    #fig.show()

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    print(fig.data[0])
    
    return graphJSON
    
    
def plot_smap(channel_name):
    
    entity = channel_name.replace("smap-", "")
    
    seed = 0
    smap = Smap_entity(seed=seed, remove_unique=False, entity = entity)
    test_df = smap.data()
  
    title = 'Original Time Series Plot of SMAP'  + entity  
    fig = px.line(test_df, x = test_df.index, y = list(test_df.columns), width=1000, height=500, title=title)
    #fig_json = fig.to_json()
    #fig.show()

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    print(fig.data[0])
    
    return graphJSON


def plot_msl(channel_name):
    entity = channel_name.replace("msl-", "")
    
    seed = 0
    msl = Msl_entity(seed=seed, remove_unique=False, entity = entity)
    test_df = msl.data()
  
    title = 'Original Time Series Plot of MSL'    
    fig = px.line(test_df, x = test_df.index, y = list(test_df.columns), width=1000, height=500, title=title)
    #fig_json = fig.to_json()
    #fig.show()

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    print(fig.data[0])
    
    return graphJSON    
    
    
def plot_smd(channel_name):
    entity = channel_name.replace("smd-", "")
    
    seed = 0
    smd = Smd_entity(seed=seed, remove_unique=False, entity = entity)
    test_df = smd.data()
  
    title = 'Original Time Series Plot of SMD'    
    fig = px.line(test_df, x = test_df.index, y = list(test_df.columns), width=1000, height=500, title=title)
    #fig_json = fig.to_json()
    #fig.show()

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    print(fig.data[0])
    
    return graphJSON    
    
    
def plot_swat():
    seed = 0
    swat = Swat(seed=seed, remove_unique=False, shorten_long=False)
    test_df = swat.data()
  
    title = 'Original Time Series Plot of SWaT'    
    fig = px.line(test_df, x = test_df.index, y = list(test_df.columns), width=1000, height=500, title=title)
    #fig_json = fig.to_json()
    #fig.show()

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    print(fig.data[0])
    
    return graphJSON    
    
    
def plot_wadi():
    seed = 0
    wadi = Wadi(seed=seed)
    test_df = wadi.data()
  
    title = 'Original Time Series Plot of WADI'    
    fig = px.line(test_df, x = test_df.Date, y = list(test_df.columns), width=1000, height=500, title=title)
    #fig_json = fig.to_json()
    #fig.show()

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    print(fig.data[0])
    
    return graphJSON    
    
def plot_damadics():
    seed = 0
    damadics = Damadics(seed=seed, drop_init_test=True)
    test_df = damadics.data()
  
    title = 'Original Time Series Plot of damadics'    
    fig = px.line(test_df, x = test_df.index, y = list(test_df.columns), width=1000, height=500, title=title)
    #fig_json = fig.to_json()
    #fig.show()

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    print(fig.data[0])
    
    return graphJSON        

#main for debugging purposes.     
#if __name__ == "__main__":
    #plot_swat()
    #plot_wadi()
    #plot_damadics()
    
    
 
