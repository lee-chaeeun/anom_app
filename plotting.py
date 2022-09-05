import plotly.graph_objects as go
import plotly.io as pio

import plotly
import plotly.express as px 

import pandas as pd
import numpy as np

import sys
import os
import json
import pickle


from timeseries_plot import plot_skab, plot_smap, plot_msl, plot_smd 

from loading import load_result, retrieve_predictions

def retrieve_plotting_configs(dataset_name,algorithm_name):    

    layout = json.dumps({'margin': dict(l=0, r=0, t=50, b=0)})
    config = json.dumps({'displaylogo': False, 'modeBarButtonsToAdd': ['drawclosedpath', 'eraseshape']})               
    header = "Visualization of Anomaly Detection Results"
    description = "Running "+dataset_name+" on " + algorithm_name 
    
    return layout, config, header, description  

def time_series(dataset_name,channel_name):
    graphJSON = []
    
    if dataset_name == 'skab':
        graphJSON = plot_skab()
    elif dataset_name == 'smap':
        graphJSON = plot_smap(channel_name)
    elif dataset_name == 'msl':
        graphJSON = plot_msl(channel_name)
    elif dataset_name == 'smd':
        graphJSON = plot_smd(channel_name)        
            
    return graphJSON    

def anomaly_scores(dataset_name,raw_predictions):
    
    columns = ['score_index','score_t_test']
   
    score_t_test_len = len( raw_predictions["score_t_test"])
    score_index = pd.DataFrame(np.arange(score_t_test_len), columns = ['score_index'])
    print("anomaly score index size is ", score_t_test_len)
    
    anomalyscore_test = pd.DataFrame(index = range(score_t_test_len), columns = columns)
    anomalyscore_test["score_index"] = score_index

    anomalyscore_test["score_t_test"] = pd.DataFrame(raw_predictions["score_t_test"], columns = ['score_t_test'])
    
    title='Test data: Anomaly Score'
    
    fig = px.line(anomalyscore_test, x = 'score_index', y = ['score_t_test'],
    width=1000, height=500, title=title)
    #fig_json = fig.to_json()
    #fig.show()

    graphJSON1 = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    print(fig.data[0])
    
    return graphJSON1


def channelwise_scores(ground_raw_predictions,raw_predictions):
    
    score_tc_test = pd.DataFrame.from_dict(raw_predictions['score_tc_test']) 
    score_tc_test.columns = ['channel ' + str(col)  for col in score_tc_test.columns]
    
    channel_num = len(score_tc_test.columns)
    print("Number of channels is", channel_num)

    
    title='Test data: Channelwise Anomaly Score'
    
    fig = px.line(score_tc_test, x = score_tc_test.index, y = list(score_tc_test.columns),
    width=1000, height=500, title=title)
    #fig_json = fig.to_json()
    #fig.show()

    graphJSON2 = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    print(fig.data[0])
    
    return graphJSON2    
    
 
def reconstruction_plot(ground_raw_predictions,raw_predictions):
    
    score_tc_test_ground = pd.DataFrame.from_dict(ground_raw_predictions['score_tc_test']) 
    score_tc_test_ground.columns = [str(col) + ' baseline' for col in score_tc_test_ground.columns]

    recons_tc_test = pd.DataFrame.from_dict(raw_predictions['recons_tc_test']) 
    recons_tc_test.columns = [str(col) + ' recons' for col in recons_tc_test.columns]
    
    predictions = pd.concat([score_tc_test_ground, recons_tc_test], axis=1)
    
    title='Baseline and Reconstruction of Channelwise Anomalies'
    
    fig = px.line(predictions, x = predictions.index, y = list(predictions.columns),
    width=1000, height=500, title=title)
    #fig_json = fig.to_json()
    #fig.show()

    graphJSON3 = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    print(fig.data[0])
    
    return graphJSON3    
    
    
def retrieve_graphs(dataset_name,ground_raw_predictions, raw_predictions):                       
    
    graphJSON1 = anomaly_scores(dataset_name,raw_predictions)
    
    graphJSON2 = channelwise_scores(ground_raw_predictions,raw_predictions)
    
    graphJSON3 = reconstruction_plot(ground_raw_predictions,raw_predictions)  
    
    return graphJSON1, graphJSON2, graphJSON3


def init_retrieve_graphs(dataset_name):

    init = pd.DataFrame(columns = ['score_t_test', 'score_tc_test','recons_tc_test'], index=range(100)) 
    init = init.fillna(0)
    
    graphJSON1 = anomaly_scores(dataset_name,init)
    
    graphJSON2 = channelwise_scores(init,init)
    
    graphJSON3 = reconstruction_plot(init,init)    
    
    return graphJSON1, graphJSON2, graphJSON3   
    

def retrieve_graphs_final(dataset_name,algorithm_name, channel_name):         

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
    
    return graphJSON1, graphJSON2, graphJSON3 


