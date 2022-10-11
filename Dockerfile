FROM continuumio/miniconda3
MAINTAINER Ryan J. McLaughlin

### EXAMPLES ###

### Build dev branch
# sudo docker build --network=host -t saber:master .

### Build test branch
# sudo docker build --build-arg git_branch=test --network=host -t saber:test .

################

Workdir /opt

### Definitions:

ENV PYTHONPATH=/opt/:/opt/libs

ARG git_branch=master

### Install apt dependencies

RUN DEBIAN_FRONTEND=noninteractive apt-get update -y 
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y python3 \
						      python3-pip \
						      wget \
						      libgmp3-dev

## Set up Conda:
COPY environment.yml /opt/environment.yml
#RUN conda update -n base -c defaults conda
RUN conda env create -f /opt/environment.yml
# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "saber_cenv", "/bin/bash", "-c"]
# Install SABer and Dependencies:
RUN pip3 install git+https://github.com/hallamlab/SABer.git@${git_branch}#egg=SABerML

# Set SABer conda env to default on startup
RUN echo "source activate saber_cenv" > ~/.bashrc
ENV PATH /opt/conda/envs/saber_cenv/bin:$PATH

## We do some umask munging to avoid having to use chmod later on,
## as it is painfully slow on large directores in Docker.
RUN old_umask=`umask` && \
    umask 0000 && \
    umask $old_umask

## Make things work for Singularity by relaxing the permissions:
RUN chmod -R 755 /opt

