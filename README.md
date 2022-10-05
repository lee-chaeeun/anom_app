

# Containerized Web Server for Visualizing Anomaly Detection of Time Series Data from CPS Systems 

## Abstract
The goal of this repository is to visualize results of anomaly detection algorithms on multivariate time-series from cyber-physical systems on a web server application in real-time. The application has a python-flask web server and is deployed in a docker container for ease of use.  This repository thereby aims to allow for future users to freely apply deep learning in various facets of industry without having expert knowledge through the development of a user-friendly web application. The respective anomaly detection application will be referred to in the following as "AnomDApp."

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

The following instructions assume that the user has access to a GPU and an Ubuntu or certain other linux distro as an OS. 

### AnomDApp Deployment in Docker

1. In the last line of environment.yml, edit prefix to show anaconda path of user pc.

2. Add datasets not included in this repository to respective paths listed below. [Datasets](https://github.com/lee-chaeeun/anom_dapp/blob/main/README.md#datasets)

3. Run following bash in terminal.

```bash
git clone https://github.com/lee-chaeeun/mvts-docker.git
docker build -t anom_dapp:latest .
docker run --gpus all -d -p 5000:5000 anomdetapp:latest
```
After building docker, an example output can be observed as such, using the `docker images` command. 
![dockerimages](/example/dockerimages.png)

After executing the run command, the terminal should respond with an output similar to the following. 
![dockerrunning](/example/dockerrun.png)

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

1. Run following bash code in terminal to create proper environment to run server

```bash
git clone https://github.com/lee-chaeeun/mvts-docker.git
conda env create -f environment.yml
source activate myenv
python3 -m pip3 install --user --upgrade pip
# Run if following packages are not downloaded properly in the environment
# based on your cuda version or use the cpu only version
#conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch 
# cpu only
# conda install pytorch==1.2.0 torchvision==0.4.0 cpuonly -c pytorch
python3 setup.py install
get_datasets.sh
```
2. Add datasets not included in this repository to respective paths listed below. [Datasets](https://github.com/lee-chaeeun/anom_dapp#datasets)

3. Run application in Conda environment
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
![Flowchart of Dataflow in AnomDApp](/example/flowchart.png)

### Server
The Flask-based web server is shown in light blue. [Flask](https://flask.palletsprojects.com/en/2.2.x/) is a micro web framework used to allow ease in development of a web server, without requiring other tools or libraries. Although it is not viable as is developed in this repository in production, it is a lightweight server that allows for quick development. Furthermore, It is regularly maintained by the "[Pallets](https://palletsprojects.com/)" community. 

The method was thereby chosen to allow for ease in development,  especially using python. This allows for ease in extension of the program from anomaly detection in the mvts-ano-eval repository on python to  the integration of said algorithms into web server development. 

### Asynchronous Task
In the diagram Flask-Executor, shown in green, is used to execute the anomaly detection algorithm as a background process to the web application in the form of a simple task queue. 

[Flask-executor](https://pypi.org/project/Flask-Executor/) is an extension package which acts as a wrapper for `concurrent.futures` module, thereby allowing users to easily launch parallel tasks on the server. Here, asynchronous execution of callables of `ThreadPoolExecutor` are wrapped by Flask-Executor with the current application context and current request context automatically.  

For the purposes of this repository, Flask-Executor executes the running of the chosen algorithm and dataset asynchronously to the rendering of the new results page and transfer of information between the server and client of AnomDApp. 

### Dataflow

The dataflow of main processes are marked with black arrows, and the changing process statuses are marked in white arrows. 

Due to the multitude of states and asynchronous tasks, a global class is declared in the application as  `app_cfg`, denoting application configuration; albeit the slight misnomer, `app_cfg.flag` is used to communicate between two main processes running parallel in the algorithm, which are the anomaly detection algorithm and the web server response to the client. 

The asynchronous task for anomaly detection outputs a string,  "start" in `app_cfg.flag` once the algorithm has began to run. 

In this case, the variable `d_flag` is used to evaluate whether or not the predictions pickle file, containing anomaly scores has been produced or not. If `d_flag` is false, the server will respond with a null output with a string, namely `ret_string` exhibiting the status, "Running Prediction." If `d_flag` is true, the server will respond with `ret_string`, containing data retrieved by functions for loading data and creating graphs within the docker working directory, as well as a string exhibiting the status message, "Running Evaluation." 

Furthermore, as soon as the server renders a new web page for the client with plot_realtime.html, an SSE connection will be opened between the client and the server such that the client may receive plotting data from the server. Here, an SSE connection refers to server-sent-event, and this is desirable for the purposes of AnomDApp because the main information exchange is that of the prediction graphs from the server to client. Once requested in accordance to control conducted by `d_flag`, the functions in loading.py are relayed onto plotting.py, where graphs for the original time-series data of the cyber-physical systems, anomaly scores, channel-wise anomaly scores, and the baseline and reconstruction of the anomaly scores are collected in the application. Here, the graphs are plotted using [Plotly](https://plotly.com/python/), an open source graphing library for python which allows for ease in production of interactive graphs exhibiting multivariate data. The `PlotlyJSONEncoder` module of `plotly.utils` is used to encode the data in JSON and export it to the client. This allows for the server to send this information through SSE connection to the client. 

Finally, the background task outputs "True" in `app_cfg.flag`, once the algorithm has run successfully. In this case, the status message string is changed to "Sucess," and the web server passes on a message in `ret_string` to the client to close the SSE connection. The server then renders the final output page with plotfinal.html. 

Once the web application page is rendered, the user may click "Return to Index page" to return to the front page of the web application. Furthermore, in the case that the user is running a dataset which has multiple channels, the different channel outputs can be observed at any time during the running of the chosen algorithm by using the drop-down menu bar provided on the upper left-hand corner of the results page. Clicking "Return to Index page" does not abruptly end the asynchronous running of the algorithm. To kill the asynchronous task, the user may either have to kill or stop the docker container or restart docker services. The option was not added to limit the control in the program managed by a global class. 

### Docker
The whole program is wrapped in a Docker container for ease of use, exhibited by the Docker logo, docker moby in blue in the upper left-hand corner of the diagram. A Docker container refers to software which is packaged using OS-level virtualization. Such a container is isolated from the rest of the device it is run on, and thereby allows for ease in downloading, running, and deleting of the software. 

In the dockerfile of this repository, the cuda libraries and conda environment profiles are downloaded on to the docker image as the docker is built. Here the Conda environment is created to run AnomDApp and all code necessary is copied on to the current working directory of the respective docker image. All necessaries packages are downloaded onto the Conda environment via environment.yml. Therefore, containerization provides the benefit that the oftentimes cumbersome tasks of GPU access and working with NVIDIA CUDA Toolkit are made easier. 

Here, [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) is a development environment including GPU-accelerated libraries which allows for faster and more efficient anomaly detection. 



## Example
Demo videos of running MSL on Autoencoder based reconstruction and SMAP on VAE-LSTM are provided in the example folder to exemplify the workings of application. The frontpage of AnomDApp is shown in the following. 
![frontpage](/example/frontpage.png)
Here the user can use a dropdown menu bar to enter the desired dataset and anomaly detection algorithm. 

If the user were to choose SMAP running on VAE-LSTM, the original time series of one channel of SMAP and the "Running Prediction" status message would show up on the results page, as exhibited in the web application shown in the right-hand side of the image. 

On the right-hand side of the image one can observe the terminal logging of the prediction exhibited by docker logs. 
![predsmapAE](/example/resultspage_running_pred_smap_VAE-LSTM.png)

Once the predictions are available, the program moves on to the evaluations step, and the status message is changed to "Running Evaluation." Finally, once evaluation is done, the program outputs "Sucesss" in the status message, and the final results page is rendered with plotfinal.html. 
![finalsmapAE](/example/resultspage_final_smap_VAE-LSTM.png)

One may observe the demo videos  [SMAP_VAE-LSTM](https://github.com/lee-chaeeun/anom_dapp/blob/cf7dd57688ec5edd258618e0428773a9080fd3f7/example/SMAP_VAE-LSTM.mkv) and [msl_AE](https://github.com/lee-chaeeun/anom_dapp/blob/cf7dd57688ec5edd258618e0428773a9080fd3f7/example/msl_AE.mkv) to better observe the application, where one may select different channels to observe the output predictions of SMAP running on VAE-LSTM and MSL running on AE. 

In the demo videos, it is important to note that the status message of "Running Evaluation" is not observable due to the program response lag when running larger datasets with multiple channels. The lightest anomaly detection algorithm and dataset to run in order to check the capabilities of AnomDApp are Autoencoder based reconstruction and SKAB.  

## Closing

### Future development, Room for improvement

System run on: 

Ubuntu 22.04 LTS
  
NVIDIA-SMI 515.65.01    Driver Version: 515.65.01    CUDA Version: 11.7   
  
NVDIDIA GeForce GTX 750 Ti with 2048 MB
 
Future systems should run on a GPU with higher RAM (maybe 8GB), CPU with storage around 300GB for dataset and results storage. 
* Current system was not able to process code for SWaT and WADI. Therefore, testing was not sufficient, though dataframes produced to exhibit graph information were successfully received as output. 
* Current system is also unable to process UAE, with terminal being killed in the process of running, with no errors being logged. 

Real-time detection responses can be improved by taking information bit by bit as it is produced before the anomaly scores are produced. A more in-depth and careful processing pipeline of this information may be useful for faster real-time output. 

Another limitation of the current application is that it is limited to running one anomaly detection algorithm with one dataset. For further ease in the user's comparison of different datasets and different algorithms, ideally the user would be able to mix and match desired combinations. Therefore, the interface could be expanded to run multiple algorithms one dataset, for example. Furthermore, another page comparing different runs could be created. 

Furthermore, with the current run, the predictions and results are saved but not used by the program again. One may take advantage of this data to be either uploaded to cloud or saved on the device such that the user may fetch the data to compare with other runs through the web interface. 

More plots could be added to display fscore and other metrics in the results file produced by the repository, mvts-ano-eval to compare different algorithms and their performance on the respective datasets. Further development  in this direction is especially recommended to better accommodate the original aim of mvts-ano-eval, which was to better compare the different techniques existing for anomaly detection. 

## Credits

### Anomaly Detection Evaluation
The above repository of AnomDApp is forked from https://github.com/astha-chem/mvts-ano-eval

### Anomaly Detection Algorithms 
All algorithms used in AnomDApp are forked from https://github.com/astha-chem/mvts-ano-eval, which uses the the following algorithms with modifications from the original sources in certain cases. 

#### PCA Reconstruction
<p>Used for lossy reconstruction </p>  
<p>S. Li and J. Wen, “A model-based fault detection and diagnostic methodology based on pca method and wavelet transform,” Energy and Buildings, vol. 68, pp. 63–71, 2014</p>

#### Univariate AutoEnconder based Reconstruction
<p>Simple fully connected univariate autoencoder, where channel-wise auto-encoders are placed for each channel. Shown in mvts-ano-eval to be best performing algorithm. </p> 
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


