import flask
from flask import Flask,render_template,request, flash, redirect, url_for, stream_with_context, request, Response
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
import os, datetime, operator, time
import timeit
import eventlet

from wtforms import Form, StringField, SelectField, validators

from plotting import anomaly_scores, time_series, channelwise_scores,reconstruction_plot

from configs import get_best_config
from src.evaluation.evaluator import analyse_from_pkls
from src.evaluation.logger_config import init_logging
from src.evaluation.trainer import Trainer
from configs import datasets_config, thres_methods, get_thres_config
from src.evaluation.evaluation_utils import get_algo_class, get_dataset_class

from celery import Celery
from celery import current_app
#from celery.bin import worker


# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# initialize the Flask application
app = flask.Flask(__name__)
app.secret_key = "super_secret_key"

app.config['CELERY_BROKER_URL'] = 'redis://127.0.0.1:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://127.0.0.1:6379/0'

celery = Celery('tasks', broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)
response = celery.control.enable_events(reply=True)

class smap_channels(Form):
    choices = [('smap-A-1', 'smap-A-1'), ('smap-A-2', 'smap-A-2'), ('smap-A-3', 'smap-A-3'), ('smap-A-4', 'smap-A-4'), ('smap-A-5', 'smap-A-5'), ('smap-A-6', 'smap-A-6'), ('smap-A-7', 'smap-A-7'), ('smap-A-8', 'smap-A-8'), ('smap-A-9', 'smap-A-9'), ('smap-B-1', 'smap-B-1'), ('smap-D-1', 'smap-D-1'), ('smap-D-11', 'smap-D-11'), ('smap-D-12', 'smap-D-12'), ('smap-D-13', 'smap-D-13'), ('smap-D-2', 'smap-D-2'), ('smap-D-3', 'smap-D-3'), ('smap-D-4', 'smap-D-4'), ('smap-D-5', 'smap-D-5'), ('smap-D-6', 'smap-D-6'), ('smap-D-7', 'smap-D-7'), ('smap-D-8', 'smap-D-8'), ('smap-D-9', 'smap-D-9'), ('smap-E-1', 'smap-E-1'), ('smap-E-10', 'smap-E-10'), ('smap-E-11', 'smap-E-11'), ('smap-E-12', 'smap-E-12'), ('smap-E-13', 'smap-E-13'), ('smap-E-2', 'smap-E-2'), ('smap-E-3', 'smap-E-3'), ('smap-E-4', 'smap-E-4'), ('smap-E-5', 'smap-E-5'), ('smap-E-6', 'smap-E-6'), ('smap-E-7', 'smap-E-7'), ('smap-E-8', 'smap-E-8'), ('smap-E-9', 'smap-E-9'), ('smap-F-1', 'smap-F-1'), ('smap-F-2', 'smap-F-2'), ('smap-F-3', 'smap-F-3'), ('smap-G-1', 'smap-G-1'), ('smap-G-2', 'smap-G-2'), ('smap-G-3', 'smap-G-3'), ('smap-G-4', 'smap-G-4'), ('smap-G-6', 'smap-G-6'), ('smap-G-7', 'smap-G-7'), ('smap-P-1', 'smap-P-1'), ('smap-P-2', 'smap-P-2'), ('smap-P-3', 'smap-P-3'), ('smap-P-4', 'smap-P-4'), ('smap-P-7', 'smap-P-7'), ('smap-R-1', 'smap-R-1'), ('smap-S-1', 'smap-S-1'), ('smap-T-1', 'smap-T-1'), ('smap-T-2', 'smap-T-2'), ('smap-T-3', 'smap-T-3')]

    select = SelectField('channel:', choices=choices)


class msl_channels(Form):
    choices = [('msl-C-1', 'msl-C-1'), ('msl-C-2', 'msl-C-2'), ('msl-D-14', 'msl-D-14'), ('msl-D-15', 'msl-D-15'), ('msl-D-16', 'msl-D-16'), ('msl-F-4', 'msl-F-4'), ('msl-F-5', 'msl-F-5'), ('msl-F-7', 'msl-F-7'), ('msl-F-8', 'msl-F-8'), ('msl-M-1', 'msl-M-1'), ('msl-M-2', 'msl-M-2'), ('msl-M-3', 'msl-M-3'), ('msl-M-4', 'msl-M-4'), ('msl-M-5', 'msl-M-5'), ('msl-M-6', 'msl-M-6'), ('msl-M-7', 'msl-M-7'), ('msl-P-10', 'msl-P-10'), ('msl-P-11', 'msl-P-11'), ('msl-P-14', 'msl-P-14'), ('msl-P-15', 'msl-P-15'), ('msl-S-2', 'msl-S-2'), ('msl-T-12', 'msl-T-12'), ('msl-T-13', 'msl-T-13'), ('msl-T-4', 'msl-T-4'), ('msl-T-5', 'msl-T-5'), ('msl-T-8', 'msl-T-8'), ('msl-T-9', 'msl-T-9')]

    select = SelectField('channel:', choices=choices)    

class smd_channels(Form):
    choices = [('smd-machine-1-1', 'smd-machine-1-1'), ('smd-machine-1-2', 'smd-machine-1-2'), ('smd-machine-1-3', 'smd-machine-1-3'), ('smd-machine-1-4', 'smd-machine-1-4'), ('smd-machine-1-5', 'smd-machine-1-5'), ('smd-machine-1-6', 'smd-machine-1-6'), ('smd-machine-1-7', 'smd-machine-1-7'), ('smd-machine-1-8', 'smd-machine-1-8'), ('smd-machine-2-1', 'smd-machine-2-1'), ('smd-machine-2-2', 'smd-machine-2-2'), ('smd-machine-2-3', 'smd-machine-2-3'), ('smd-machine-2-4', 'smd-machine-2-4'), ('smd-machine-2-5', 'smd-machine-2-5'), ('smd-machine-2-6', 'smd-machine-2-6'), ('smd-machine-2-7', 'smd-machine-2-7'), ('smd-machine-2-8', 'smd-machine-2-8'), ('smd-machine-2-9', 'smd-machine-2-9'), ('smd-machine-3-1', 'smd-machine-3-1'), ('smd-machine-3-10', 'smd-machine-3-10'), ('smd-machine-3-11', 'smd-machine-3-11'), ('smd-machine-3-2', 'smd-machine-3-2'), ('smd-machine-3-3', 'smd-machine-3-3'), ('smd-machine-3-4', 'smd-machine-3-4'), ('smd-machine-3-5', 'smd-machine-3-5'), ('smd-machine-3-6', 'smd-machine-3-6'), ('smd-machine-3-7', 'smd-machine-3-7'), ('smd-machine-3-8', 'smd-machine-3-8'), ('smd-machine-3-9', 'smd-machine-3-9')]

    select = SelectField('channel:', choices=choices)  
    
class no_channels(Form):
    choices = [('','N/A')]
    select = SelectField('channel:', choices=choices)   

@celery.task(bind=True)                      
def run_algorithm(self,dataset_name, algorithm_name):  

    self.update_state(state='STARTED')
    out_dir_root = os.path.join(os.getcwd(), "reports", "trial")
    multi_seeds = [0]
    ds_to_run = [dataset_name]
    algos_to_run = [
                    "RawSignalBaseline",
                    algorithm_name
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

                trainer.train_predict()
                
                self.update_state(state='RUNNING')
                          
                analyse_from_pkls(results_root=out_dir_algo, thres_methods=thres_methods, eval_root_cause=True, point_adjust=False,
                                  eval_dyn=True, eval_R_model=True, thres_config=get_thres_config,
                                  telem_only=True, composite_best_f1=True)    

                                   

            
def newest(directory):
    files = os.listdir(directory)
    all_subdirs = [os.path.join(directory,basename) for basename in files if os.path.isdir(os.path.join(directory,basename))]
    newest = max(all_subdirs, key = os.path.getctime)
    return newest

def load_smap(run_path):
    files = os.listdir(run_path)
    all_subdirs = [os.path.join(directory,basename) for basename in files if os.path.isdir(os.path.join(directory,basename))]
    newest = max(all_subdirs, key = os.path.getctime)
    return newest
    
def load_result(data_name, algo_name):
    out_dir_root = os.path.join(os.getcwd(), "reports", "trial") 
    out_dir_algo = os.path.join(out_dir_root, algo_name) #reports/trial/algo_name
    data_path = data_name + "_me" #data_me 
    if algo_name == "VAE-LSTM":
        search_path = os.path.join(out_dir_algo, data_path, "VAE_LSTM") #reports/trial/algo_name/data_me/algo_name
    else:     
        search_path = os.path.join(out_dir_algo, data_path, algo_name) #reports/trial/algo_name/data_me/algo_name
    base_config = newest(search_path) #data_me/algo_name/base-configxxx  
    entity_path = newest(base_config) #data_me/algo_name/base-configxxx/0-run-xxx     
    return entity_path  


         
# A decorator used to tell the application
# which URL is associated function
@app.route('/', methods =["GET", "POST"])
def user_input():
    if request.method == "POST":
       # getting input with name = fname in HTML form
       dataset_name = request.form.get("datasets_dropdown")
       # getting input with name = lname in HTML form
       algorithm_name = request.form.get("algorithms_dropdown")
       
       #Run multi-variate time-series anomaly detection
       try:     
           #run anomaly detection in background using celery
           task = run_algorithm.apply_async(args=[dataset_name,algorithm_name])
           print_on_flask = "Running "+dataset_name+" on " + algorithm_name
           flash(print_on_flask)
           print(print_on_flask)
           return redirect(url_for('results', dataset_name = dataset_name, algorithm_name = algorithm_name,task_id=task.id)) 
           
       except Exception as e:
           print_on_flask = "Error running "+dataset_name+" on " + algorithm_name
           flash(print_on_flask)
           print(print_on_flask)
           print(e)
           return redirect(url_for('error', dataset_name = dataset_name, algorithm_name = algorithm_name)) 
           
    datasets_dropdown = ['msl', 'smap', 'smd', 'skab']
    algorithms_dropdown = ['PcaRecons', 'UnivarAutoEncoder_recon_all', 'AutoEncoder_recon_all', 'LSTM-ED_recon_all', 'TcnED', 'VAE-LSTM', 'MSCRED', 'OmniAnoAlgo']
       
    return render_template("form.html", datasets_dropdown = datasets_dropdown, algorithms_dropdown = algorithms_dropdown)
    
    
@app.route('/results/<dataset_name>/<algorithm_name>/<task_id>', methods=['GET', 'POST'])
def results(dataset_name,algorithm_name,task_id):
    dataset_name = dataset_name
    algorithm_name = algorithm_name
        
    if dataset_name == "smap":
        channel_name = dataset_name+"-A-1"  
        #for channel-dependent plots 
        channel_numbers = smap_channels(request.form)
     
    elif dataset_name == "msl":
        channel_name = dataset_name+"-C-1"   
        channel_numbers = msl_channels(request.form) 
        
    elif dataset_name == "smd":
        channel_name = dataset_name+"-machine-1-1"
        channel_numbers = smd_channels(request.form) 
        
    else:
        channel_name = dataset_name 
        channel_numbers = no_channels(request.form)   

    if request.method == 'POST':
        channel_name = channel_numbers.data['select']

    task = run_algorithm.AsyncResult(task_id)
    if task.state == 'RUNNING':
        #Process(target=visualize,args=(dataset_name,algorithm_name,channel_name,q)).start()
        data, response = visualize(dataset_name,algorithm_name, channel_name)
    else: 
        data = { "graphJSON1" : np.zeros(1),"graphJSON2" : np.zeros(1),"graphJSON3" : np.zeros(1)}    
     
    graphJSON1 = data["graphJSON1"]
    graphJSON2 = data["graphJSON2"]
    graphJSON3 = data["graphJSON3"]    
    
    layout = json.dumps({'margin': dict(l=0, r=0, t=50, b=0)})
    config = json.dumps({'displaylogo': False, 'modeBarButtonsToAdd': ['drawclosedpath', 'eraseshape']})   
            
    header = "Visualization of Anomaly Detection Results"
    description = "Running "+dataset_name+" on " + algorithm_name   
            
    #graph og time series
    graphJSON = time_series(dataset_name,channel_name)

    return render_template("plot.html", graphJSON=graphJSON, graphJSON1=graphJSON1, graphJSON2=graphJSON2, graphJSON3=graphJSON3, layout=layout, config=config, header = header, description = description, form = channel_numbers)     

 
#@app.route('/results/<dataset_name>/<algorithm_name>')    
def visualize(dataset_name,algorithm_name, channel_name):
           
    try: 
        #load Raw Base Signal for ground truth      
        groundtruthpath = os.path.join(load_result(dataset_name, "RawSignalBaseline"), channel_name)       
        #anomaly scores ground truth 
        with open(os.path.join(groundtruthpath, "raw_predictions"),'rb') as file:
           ground_raw_predictions = pickle.load(file)
    except Exception as e:    
        print(e)
        ground_raw_predictions = np.zeros(1)
        
    try:    
        #load raw predicitions from algorithm chosen     
        resultpath = os.path.join(load_result(dataset_name, algorithm_name), channel_name)        
        #anomaly scores 
        with open(os.path.join(resultpath, "raw_predictions"),'rb') as file:
           raw_predictions = pickle.load(file)
    except Exception as e:    
        print(e)  
        raw_predictions =  np.zeros(1)
    
    graphJSON1 = anomaly_scores(dataset_name,raw_predictions)
    
    graphJSON2 = channelwise_scores(ground_raw_predictions,raw_predictions)
    
    graphJSON3 = reconstruction_plot(ground_raw_predictions,raw_predictions)     
   
    
    #q.put({"graphJSON1" : graphJSON1,"graphJSON2" : graphJSON2,"graphJSON3" : graphJSON3 })
        
    data = {
            "graphJSON1" : graphJSON1,
            "graphJSON2" : graphJSON2,
            "graphJSON3" : graphJSON3,       
        }       
        
    #Process(target=run_algorithm,args=(dataset_name,algorithm_name)).start()
    
    return data, Response(stream_with_context(data), mimetype="text/event-stream")    
    
    
        
@app.route('/error/<dataset_name>/<algorithm_name>')
def error(dataset_name,algorithm_name):
    dataset_name = dataset_name
    algorithm_name = algorithm_name
    
    header = "Error running "+dataset_name+" on " + algo.rithm_name
    description = "check terminal for error sources"
    
    #graph og time series
    graphJSON = time_series(dataset_name)
    
    layout = json.dumps({'margin': dict(l=0, r=0, t=50, b=0)})
    config = json.dumps({'displaylogo': False, 'modeBarButtonsToAdd': ['drawclosedpath', 'eraseshape']})
    
    return render_template("error.html", graphJSON=graphJSON, layout=layout, config=config, header = header, description = description)    

        
# first load the model and then start the server
# we need to specify the host of 0.0.0.0 so that the app is available on both localhost as well
# as on the external IP of the Docker container
if __name__ == "__main__":

    app.run(host='0.0.0.0', port=8080)
    
    argv = [
        'worker',
        '--loglevel=info',
        '-f celery.logs',
        '--without-gossip',     
        '--pool=eventlet',          
    ]
    celery.worker_main(argv)
    print(response)

