#FROM continuumio/miniconda3
FROM nvidia/cuda:10.0-cudnn7-runtime

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update 

RUN echo '#! /bin/sh' > /usr/bin/mesg
RUN chmod 755 /usr/bin/mesg
    
# INSTALLATION OF libraries
RUN apt-get install -y wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion && \
        apt-get clean
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh -O ~/anaconda.sh && \
        /bin/bash ~/anaconda.sh -b -p /opt/conda && \
        rm ~/anaconda.sh && \
        ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
        echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc 
    
# set path to conda
ENV PATH /opt/conda/bin:$PATH

# Create the environment
RUN conda update conda
#remove old env
RUN conda remove --name myenv --all
#create new env
COPY environment.yml .
RUN conda env create -f environment.yml

# Make RUN commands use the new environment
RUN echo "conda activate myenv" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# You can add the new created environment to the path
ENV PATH /opt/conda/envs/myenv/bin:$PATH

WORKDIR /app

#copy the contents of the current working directory. 
#copies them into the image.
COPY README.md app.py channel_classes.py loading.py plotting.py timeseries_plot.py setup.py configs.py /app/
COPY data/ /app/data/
COPY src/ /app/src/
COPY templates/ /app/templates/

#RUN ls -la /app/*

#ENV FLASK_DEBUG=1
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
EXPOSE 5000

# The code to run when container is started:
COPY app.py entrypoint.sh ./
RUN chmod +x entrypoint.sh
CMD ["./entrypoint.sh"]
