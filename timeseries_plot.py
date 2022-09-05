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
    
