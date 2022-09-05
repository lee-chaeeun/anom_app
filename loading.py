import os, datetime, operator, time, pickle
from datetime import datetime

def newest(directory):
    files = os.listdir(directory)
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
    
def retrieve_predictions(dataset_name,algorithm_name, channel_name, c_time):
                       
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

        d_time = datetime.fromtimestamp(os.path.getctime(resultpath))   
        print("data created:",d_time,"\n","code started:", c_time)     
        
        if d_time < c_time: 
            print("not yet time, my young grasshopper")
            d_flag = False
        else: 
            print("raw predictions available")
            d_flag = True    
           
    except Exception as e:    
        print(e)  
        raw_predictions =  np.zeros(1)
        
    return ground_raw_predictions, raw_predictions, d_flag
