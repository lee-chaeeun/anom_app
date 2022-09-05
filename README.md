## Containerized Anomaly Detection on Time Series Visualized on Web Server
The goal of this repository is to visualize results of multivariate time-series anomaly detection algorithms discussed in the paper on a python flask web server and to deploy the server in a docker container. 
***Repository is still in production***

The following files were added to plot real-time results on a flask server. \n
app.py \n
loading.py
channel_classes.py
plotting.py
timeseries_plot.py

The respective templates to the flask code are located in the templates folder. 
form.html
_formhelpers.html
error.html
plot_final.html
plot_realtime.html

```bash
git clone https://github.com/lee-chaeeun/mvts-docker.git
conda env create -f environment.yml
source activate mvtsenvs
python3 setup.py install
get_datasets.sh
python3 app.py
```

## Credits
This repository is forked from https://github.com/astha-chem/mvts-ano-eval
