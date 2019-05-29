FROM continuumio/anaconda3

RUN useradd jupyter --create-home

RUN conda install -y keras
RUN conda install -y nltk
RUN conda install -y scikit-learn

USER jupyter

RUN cd

RUN jupyter notebook --generate-config

RUN conda -V
