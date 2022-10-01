#for FLASK APP and BACKGROUND TASK
import flask
from flask import Flask,render_template,request, flash, redirect, url_for, request, Response, stream_with_context
import requests
from flask_executor import Executor

import plotly as py
import plotly.express as px 
import plotly.graph_objs as go

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
from datetime import datetime

#to RUN ANOMALY DETECTION 
from configs import get_best_config
from src.evaluation.evaluator import analyse_from_pkls
from src.evaluation.logger_config import init_logging
from src.evaluation.trainer import Trainer
from configs import datasets_config, thres_methods, get_thres_config
from src.evaluation.evaluation_utils import get_algo_class, get_dataset_class

#to REAL TIME PLOT 
from channel_classes import get_channel_info
from loading import retrieve_predictions
from plotting import time_series, retrieve_graphs, retrieve_plotting_configs, init_retrieve_graphs, retrieve_graphs_final
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# initialize the Flask application
app = flask.Flask(__name__)
app.secret_key = "super_secret_key"

app.config['EXECUTOR_PROPAGATE_EXCEPTIONS'] = True

executor = Executor(app)


#-------------------------------
# Configuration of the application.
# flag: to tell if algorithm has finished producing predictions or not
# dataset_name
# algorithm_name
# c_time: time code has started to run to compare with d_time, time predictions are done. 
#-------------------------------
class Config: 
    flag = False   
    dataset_name = ''
    algorithm_name = ''    
    c_time = 0
    
# Instantiate app_config
app_cfg = Config

#---------------------------------------
# App Routes
#---------------------------------------

@executor.job             
def run_algorithm(dataset_name, algorithm_name):  
  
    out_dir_root = os.path.join(os.getcwd(), "reports", "trial")
    multi_seeds = [0]
    ds_to_run = [dataset_name]
    algos_to_run = [
                    "RawSignalBaseline",
                    algorithm_name,
                    ]
    test_run=True
    
    #def run_multi_seeds(out_dir_root, multi_seeds, ds_to_run, algos_to_run, test_run=False):                    
    for seed in multi_seeds:
        for ds_name in ds_to_run:
            for algo_name in algos_to_run:
                algo_config_dict = get_best_config(algo_name=algo_name, ds_name=ds_name)
                if test_run:
                    if "num_epochs" in algo_config_dict.keys():
                        algo_config_dict["num_epochs"] = 1
                out_dir_algo = os.path.join(out_dir_root, algo_name)
                
                #def train_analyse_algo(ds_name, algo_name, algo_config_dict, out_dir_algo, seed):                
                init_logging(os.path.join(out_dir_algo, 'logs'))
                logger = logging.getLogger(__name__)
                if ds_name in datasets_config.keys():
                    ds_kwargs = datasets_config[ds_name]
                else:
                    ds_kwargs = {}
                trainer = Trainer(ds_class=get_dataset_class(ds_name),
                                  algo_seeds=[seed],
                                  algo_class=get_algo_class(algo_name),
                                  ds_seed=seed,
                                  ds_kwargs=ds_kwargs,
                                  algo_config_base=algo_config_dict,
                                   output_dir=out_dir_algo,
                                  logger=logger)
                print(
                "Training algo {} on dataset {} with config {} and seed {}".format(algo_name, ds_name, algo_config_dict, seed))
                
                app_cfg.flag = 'start'                
                trainer.train_predict()                                                
                         
                analyse_from_pkls(results_root=out_dir_algo, thres_methods=thres_methods, eval_root_cause=True, point_adjust=False,
                                  eval_dyn=True, eval_R_model=True, thres_config=get_thres_config,
                                  telem_only=True, composite_best_f1=True)    
    app_cfg.flag = True    
                                  

@app.route('/', methods =["GET", "POST"])
def user_input():
    if request.method == "POST":
       # getting input with name = fname in HTML form
       dataset_name = request.form.get("datasets_dropdown")
       # getting input with name = lname in HTML form
       algorithm_name = request.form.get("algorithms_dropdown")
       
       #Run multi-variate time-series anomaly detection
       try:     
           #run anomaly detection in background
           print_on_flask = "Running "+dataset_name+" on " + algorithm_name
           #flash(print_on_flask)
           print(print_on_flask)
           
           run_algorithm.submit(dataset_name, algorithm_name)
           
           channel_numbers, channel_name = get_channel_info(dataset_name)
           
           return redirect(url_for('results', dataset_name = dataset_name, algorithm_name = algorithm_name)) 
           
       except Exception as e:
           print_on_flask = "Error running "+dataset_name+" on " + algorithm_name
           flash(print_on_flask)
           print(print_on_flask)
           print(e)
           return redirect(url_for('error', dataset_name = dataset_name, algorithm_name = algorithm_name)) 
           
    datasets_dropdown = ['msl', 'smap', 'smd', 'skab', 'swat', 'wadi', 'damadics-s']  
    algorithms_dropdown = ['PcaRecons', 'UnivarAutoEncoder_recon_all', 'AutoEncoder_recon_all', 'LSTM-ED_recon_all', 'TcnED', 'VAE-LSTM', 'MSCRED', 'OmniAnoAlgo']
       
    return render_template("index.html", datasets_dropdown = datasets_dropdown, algorithms_dropdown = algorithms_dropdown)
    

 
@app.route('/results/<dataset_name>/<algorithm_name>/', methods=['GET', 'POST'])
def results(dataset_name,algorithm_name):

    channel_numbers, channel_name = get_channel_info(dataset_name)
    
    if request.method == 'POST':
        channel_name = channel_numbers.data['select']

    layout, config, header, description = retrieve_plotting_configs(dataset_name,algorithm_name)
         
    app_cfg.dataset_name = dataset_name
    app_cfg.algorithm_name = algorithm_name
    app_cfg.c_time = datetime.now()
    
    #graph og time series
    graphJSON = time_series(dataset_name,channel_name)      
    
    #if streaming streaming/running over
    if app_cfg.flag is True:   
        graphJSON1, graphJSON2, graphJSON3 = retrieve_graphs_final(dataset_name,algorithm_name, channel_name)         
        return render_template("plot_final.html", graphJSON=graphJSON, graphJSON1= graphJSON1, graphJSON2=graphJSON2, graphJSON3=graphJSON3,layout=layout, config=config, header = header, description = description, form = channel_numbers)
        
    return render_template("plot_realtime.html", graphJSON=graphJSON, layout=layout, config=config, header = header, description = description, form = channel_numbers)       
          
          
@app.route('/progress/')
def progress():
            
    @stream_with_context
    def generate():
        x = 0
        dataset_name = app_cfg.dataset_name  
        algorithm_name = app_cfg.algorithm_name     
        channel_numbers, channel_name = get_channel_info(dataset_name)    
        update_rate = 5
        ret_dict = {} 
        d_flag = False      
                       
        while app_cfg.flag != True :       
          
            print("app_cfg.flag: ", app_cfg.flag)
                    
            while app_cfg.flag is 'start':
                
                ret_dict[1] = "Running prediction"
                 
                ground_raw_predictions, raw_predictions, d_flag =retrieve_predictions(dataset_name,algorithm_name, channel_name, app_cfg.c_time) 
                
                #if d_flag == False: raw predictions not yet available
                graphJSON1, graphJSON2, graphJSON3 = init_retrieve_graphs(dataset_name)
                
                #if d_flag true, thus raw predictions generated 
                if d_flag == True:     
                    ret_dict[1] = "Running evaluation"     
                    graphJSON1, graphJSON2, graphJSON3 = retrieve_graphs(dataset_name,ground_raw_predictions, raw_predictions) 
          
                ret_dict[2] = graphJSON1
                ret_dict[3] = graphJSON2            
                ret_dict[4] = graphJSON3            
            
                ret_string = "data:" + json.dumps(ret_dict) + "\n\n"        
                
                yield ret_string
                                                                                                      
                time.sleep(update_rate)  
                
        else :
        
            ret_dict[1] = "Sucess"  
            
            graphJSON1, graphJSON2, graphJSON3 = retrieve_graphs_final(dataset_name,algorithm_name, channel_name) 
            ret_dict[2] = graphJSON1
            ret_dict[3] = graphJSON2            
            ret_dict[4] = graphJSON3   
                                   
            ret_string = "data:" + json.dumps(ret_dict) + "\n\n"     
                      
            yield ret_string                    
                                 
    return Response(generate(), mimetype= 'text/event-stream')   
         
            
# first load the model and then start the server
# we need to specify the host of 0.0.0.0 so that the app is available on both localhost as well
# as on the external IP of the Docker container
if __name__ == "__main__":

    app.run(host='0.0.0.0', port=5000, debug=False)   
    

