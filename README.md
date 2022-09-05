## Containerized Anomaly Detection on Time Series Visualized on Web Server
The goal of this repository is to visualize results of multivariate time-series anomaly detection algorithms discussed in the paper on a python flask web server and to deploy the server in a docker container. 
***Repository is still in production***

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
