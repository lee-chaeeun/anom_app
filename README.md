# Containerized Anomaly Detection on Time Series Visualized on Web Server
The goal of this repository is to real-time visualize results of multivariate time-series anomaly detection algorithms discussed in the paper on a python flask web server and to deploy the server in a docker container. The anomaly detection application will be referred to in the following as "AnomApp."

***Repository is still in production***

The following files were added to plot real-time results on a flask server. 
* app.py 
* loading.py
* channel_classes.py
* plotting.py
* timeseries_plot.py
* dockerfile
* entrypoint.sh
* environment.yml

The respective templates to the flask code are located in the templates folder. 
* form.html
* _formhelpers.html
* error.html
* plot_final.html
* plot_realtime.html

The following src.dataset code was slightly edited to fit run_algorithm in app.py
* wadi.py
* damadics.py
* swat.py

**Table of Contents**

[TOC]

## Instructions to run

### Run AnomApp in Docker

1. In the last line of environment.yml, edit prefix to show anaconda path of user pc.

2. Add datasets not included in this repository to respective paths listed below. [Datasets](https://github.com/anom_app/README.md "Datasets")

3. Run following bash in terminal.

```bash
git clone https://github.com/lee-chaeeun/mvts-docker.git
docker build -t anom_app:latest .
docker run --gpus all -d -p 5000:5000 anomdetapp:latest
```
***Tips for debugging in future production***

Show error or logs of docker after running 

`docker logs <container number>`

Show active docker containers 

`docker ps`

Show all active and non-active docker containers 

`docker ps -a`

Stop active docker container 

`docker container stop your_container_id`

Options to try if docker container is not stopping and an error exists

`sudo docker container kill`

`sudo systemctl restart docker.socket docker.service`

Fix if there is an "already in use" error for port

`sudo netstat -pna | grep 5000`

`sudo kill <process id>`

Delete all containers and images 

`sudo docker image prune -a --force --filter "until=2022-10-07T10:00:00"`

`sudo docker system prune -a`

### Run AnomApp in Conda environment 

1. Run following bash code in terminal 

```bash
git clone https://github.com/lee-chaeeun/mvts-docker.git
conda env create -f environment.yml
source activate mvtsenvs
python3 -m pip3 install --user --upgrade pip
# based on your cuda version or use the cpu only version
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch 
# cpu only
# conda install pytorch==1.2.0 torchvision==0.4.0 cpuonly -c pytorch
python3 setup.py install
get_datasets.sh
```
2. Add datasets not included in this repository to respective paths listed below. [Datasets](https://github.com/anom_app/README.md "Datasets")

```bash
python3 app.py
```
***Tips for debugging in future production***
This option is more volatile compared to running docker; one must pay careful attention to version mismatches in environment of each package. 

## Datasets

### Skab
source: [website](https://www.kaggle.com/yuriykatser/skoltech-anomaly-benchmark-skab/version/1)
set path: <root-of-the-project>/data/raw/skab

### Smap and Msl

source:
```bash
wget https://s3-us-west-2.amazonaws.com/telemanom/data.zip && unzip data.zip && rm data.zip
cd data && wget https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv
```
set path: <root-of-the-project>/data/raw/smap_msl

### Damadics
source: [website](http://diag.mchtr.pw.edu.pl/damadics/)
set path: <root-of-the-project>/data/raw/damadics/raw

### Smd
source: (Server Machine Dataset) [website](https://github.com/NetManAIOps/OmniAnomaly/tree/master/ServerMachineDataset)
set path: <root-of-the-project>/data/raw/ServerMachineDataset 

### Swat
source: fill out following form from iTrust
[this form](https://docs.google.com/forms/d/e/1FAIpQLSfnbjv7ZnDNmV_5ge7OfUc_O_h5yUnj708TFL8dD3o3Yoj9Fw/viewform)

Download the files "SWaT_Dataset_Normal_v1.xlsx" and "SWaT_Dataset_Attack_v0.xlsx" 
set path: <root-of-the-project>/data/raw/swat/raw

### Wadi
source: fill out following form from iTrust
[form](https://docs.google.com/forms/d/e/1FAIpQLSfnbjv7ZnDNmV_5ge7OfUc_O_h5yUnj708TFL8dD3o3Yoj9Fw/viewform) 
take csv tables in 02_WADI Dataset_19 Nov 2017 folder
rename anomalies file to WADI_anomalies
set path: <root-of-the-project>/data/raw/wadi/raw

## Future development, Room for improvement 

System run on: 
  
NVIDIA-SMI 515.65.01    Driver Version: 515.65.01    CUDA Version: 11.7   
  
NVDIDIA GeForce GTX 750 Ti with 2048 MB
 
Future systems should run on a GPU with higher RAM (maybe 8GB), CPU with storage around 300GB for dataset and results storage. 
* current system was not able to process code for SWaT and WADI. Therefore, testing was not sufficient, though dataframes produced to exhibit graph information were successfully received as output. 
* Current system is also unable to process UAE, with terminal being killed in the process of running, with no error. 

Real-time detection responses can be improved by taking information bit by bit as it is produced before the anomaly scores are produced. A more in-depth processing pipeline of this information may be useful for faster real-time output. 

More plots could be added to display fscore and other metrics to compare different algorithms and their performance on the respective datasets. 


## Credits
This repository is forked from https://github.com/astha-chem/mvts-ano-eval

