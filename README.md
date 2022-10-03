
# Containerized Anomaly Detection on Time Series Visualized on Web Server

## Abstract
The goal of this repository is to visualize results of anomaly detection algorithms on multivariate time-series  discussed in the paper on a REST API in real-time. The REST API is a flask web server and is deployed in a docker container for ease of use. The anomaly detection application will be referred to in the following as "AnomDApp."

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

1. [Usage](https://github.com/lee-chaeeun/anom_dapp/blob/main/README.md#usage)

   *  [AnomDApp Deployment in Docker](https://github.com/lee-chaeeun/anom_dapp/blob/main/README.md#anomdapp-deployment-in-docker)
   *  [AnomDApp in Conda environment](https://github.com/lee-chaeeun/anom_dapp/blob/main/README.md#run-anomdapp-in-conda-environment)  

2. [Datasets](https://github.com/lee-chaeeun/anom_dapp/blob/main/README.md#datasets)

3. [AnomDApp Algorithm](https://github.com/lee-chaeeun/anom_dapp/blob/main/README.md#anomdapp-algorithm) 

4. [Example](https://github.com/lee-chaeeun/anom_dapp/blob/main/README.md#example)

5.  [Future development, Room for improvement ](https://github.com/lee-chaeeun/anom_dapp/blob/main/README.md#future-development-room-for-improvement)

6. [Credits](https://github.com/lee-chaeeun/anom_dapp/blob/main/README.md#credits)

   *  [Anomaly Detection Evaluation](https://github.com/lee-chaeeun/anom_dapp/blob/main/README.md#anomaly-detection-evaluation)
   *  [Anomaly Detection Algorithms](https://github.com/lee-chaeeun/anom_dapp#anomaly-detection-algorithms)


## Usage

### AnomDApp Deployment in Docker

1. In the last line of environment.yml, edit prefix to show anaconda path of user pc.

2. Add datasets not included in this repository to respective paths listed below. [Datasets](https://github.com/anom_dapp/README.md "Datasets")

3. Run following bash in terminal.

```bash
git clone https://github.com/lee-chaeeun/mvts-docker.git
docker build -t anom_dapp:latest .
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

### Run AnomDApp in Conda environment 

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
2. Add datasets not included in this repository to respective paths listed below. [Datasets](https://github.com/lee-chaeeun/anom_dapp#datasets)

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
set path: \<root-of-the-project>/data/raw/smap_msl

### Damadics
source: [website](https://iair.mchtr.pw.edu.pl/Damadics)
set path: \<root-of-the-project>/data/raw/damadics/raw

### Smd
source: (Server Machine Dataset) [website](https://github.com/NetManAIOps/OmniAnomaly/tree/master/ServerMachineDataset)
set path: \<root-of-the-project>/data/raw/ServerMachineDataset 

### Swat
source: fill out following form from iTrust
[this form](https://docs.google.com/forms/d/e/1FAIpQLSfnbjv7ZnDNmV_5ge7OfUc_O_h5yUnj708TFL8dD3o3Yoj9Fw/viewform)

Download the files "SWaT_Dataset_Normal_v1.xlsx" and "SWaT_Dataset_Attack_v0.xlsx" 
set path: \<root-of-the-project>/data/raw/swat/raw

### Wadi
source: fill out following [form](https://docs.google.com/forms/d/e/1FAIpQLSfnbjv7ZnDNmV_5ge7OfUc_O_h5yUnj708TFL8dD3o3Yoj9Fw/viewform)  from iTrust

take csv tables in 02_WADI Dataset_19 Nov 2017 folder
rename anomalies file to WADI_anomalies
set path: \<root-of-the-project>/data/raw/wadi/raw

## AnomDApp Algorithm
![Flowchart of Dataflow in AnomDApp](flowchart.png)

Flask-Executor is used to execute background processes. 

## Example
Demo videos of running Skab, MSL, and SMAP on an Autoencoder as well as example of AnomDApp are provided in the example folder. The frontpage of AnomDApp is shown in the following. 
![frontpage](/example/frontpage.png)
Here the user can use a dropdown menu bar to enter the desired dataset and anomaly detection algorithm. 

If the user were to choose SMAP running on VAE-LSTM, the original time series of one channel of SMAP and the "Running Prediction" status message would show up on the results page, as exhibited in the web application shown in the right-hand side of the image. 

On the right-hand side of the image one can observe the terminal logging of the prediction exhibited by docker logs.  GPU was not used in this case. 
![predsmapAE](/example/resultspage_running_pred_smap_VAE-LSTM.png)

Once the predictions are available, the program moves on to the evaluations step, and the status message is changed to "Running Evaluation." Finally, once evaluation is done, the program outputs "Sucesss" in the status message, and the final results page is rendered with plotfinal.html.
![finalsmapAE](/example/resultspage_final_smap_VAE-LSTM.png)

One may observe the demo videos  [SMAP_VAE-LSTM](https://github.com/lee-chaeeun/anom_dapp/example/blob/main/SMAP_AE.mkv) and [msl_AE](https://github.com/lee-chaeeun/anom_dapp/example/blob/main/msl_AE) to better observe the application, where one may select different channels to observe the output predictions of SMAP and MSL running on AE. 



## Closing

### Future development, Room for improvement

System run on: 
  
NVIDIA-SMI 515.65.01    Driver Version: 515.65.01    CUDA Version: 11.7   
  
NVDIDIA GeForce GTX 750 Ti with 2048 MB
 
Future systems should run on a GPU with higher RAM (maybe 8GB), CPU with storage around 300GB for dataset and results storage. 
* current system was not able to process code for SWaT and WADI. Therefore, testing was not sufficient, though dataframes produced to exhibit graph information were successfully received as output. 
* Current system is also unable to process UAE, with terminal being killed in the process of running, with no error. 

Real-time detection responses can be improved by taking information bit by bit as it is produced before the anomaly scores are produced. A more in-depth processing pipeline of this information may be useful for faster real-time output. 

More plots could be added to display fscore and other metrics to compare different algorithms and their performance on the respective datasets. 


## Credits

### Anomaly Detection Evaluation
The above repository of AnomDApp is forked from https://github.com/astha-chem/mvts-ano-eval

### Anomaly Detection Algorithms 
All algorithms used in AnomDApp are forked from https://github.com/astha-chem/mvts-ano-eval, which uses the the following algorithms with modifications from the original sources in certain cases. 

#### PCA Reconstruction
<p>Used for lossy reconstruction </p>  
<p>S. Li and J. Wen, “A model-based fault detection and diagnostic methodology based on pca method and wavelet transform,” Energy and Buildings, vol. 68, pp. 63–71, 2014</p>

#### Univariate AutoEnconder based Reconstruction
<p>Simple fully connected univariate autoencoder, where channel-wise auto-encoders are placed for each channel. Shown in mvts-ano-eval to be best performing algorithm </p> 
<p>requires high computation power and GPU memory to save predictions and evaluation. </p>  

#### AutoEncoder based Reconstruction

<p>S. Hawkins, H. He, G. Williams, and R. Baxter, “Outlier detection using replicator neural networks,” in International Conference on Data Warehousing and Knowledge Discovery. Springer, 2002, pp. 170–180.</p>

#### LSTM-ED based Reconstruction

<p>P. Malhotra, A. Ramakrishnan, G. Anand, L. Vig, P. Agarwal, and G. Shroff, “Lstm-based encoder-decoder for multi-sensor anomaly detection,” arXiv preprint arXiv:1607.00148, 2016.</p>

#### TcnED 
<p>based on work from following paper<p/> 
<p> S. Bai, J. Z. Kolter, and V. Koltun, “An empirical evaluation of generic convolutional and recurrent networks for sequence modeling,” arXiv preprint arXiv:1803.01271, 2018.</p>

#### VAE-LSTM 

<p>based on work from following paper<p/> 
<p>D. Park, Y. Hoshi, and C. C. Kemp, “A multimodal anomaly detector for robot-assisted feeding using an lstm-based variational autoencoder,” IEEE Robotics and Automation Letters, vol. 3, no. 3, pp. 1544–1551, 2018.</p>        

#### MSCRED
<p>requires high computation power and GPU memory. </p>        
<p> C. Zhang, D. Song, Y. Chen, X. Feng, C. Lumezanu, W. Cheng, J. Ni, B. Zong, H. Chen, and N. V. Chawla, “A deep neural network for unsupervised anomaly detection and diagnosis in multivariate time series data,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 33, 2019, pp. 1409–1416.</p>    

#### OmniAnoAlgo

<p>Y. Su, Y. Zhao, C. Niu, R. Liu, W. Sun, and D. Pei, “Robust anomaly detection for multivariate time series through stochastic recurrent neural network,” in Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, 2019, pp. 2828–
2837.</p>   


