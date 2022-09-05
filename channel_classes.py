from wtforms import Form, StringField, SelectField, validators
from flask import Flask, request 

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
    
    
def get_channel_info(dataset_name):

    #get CHANNEL info
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
        channel_numbers = no_channels()  
    
    return channel_numbers, channel_name    
    
    
