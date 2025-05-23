FROM nvidia/cuda:10.1-devel

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
  noweb libcurl4-openssl-dev \
    git wget bzip2 ca-certificates

# RUN wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh
# RUN bash Miniconda-latest-Linux-x86_64.sh -p /miniconda -b
# RUN rm Miniconda-latest-Linux-x86_64.sh
# ENV PATH=/miniconda/bin:${PATH}
# RUN conda update -y conda

ENV PATH /home/anaconda/bin:$PATH

RUN wget --quiet https://repo.continuum.io/archive/Anaconda3-2019.10-Linux-x86_64.sh -O ~/anaconda.sh && \
                 /bin/bash ~/anaconda.sh -b -p /home/anaconda && \                                
                    rm ~/anaconda.sh && \
                    ln -s /home/anaconda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
                    echo ". /home/anaconda/etc/profile.d/conda.sh" >> ~/.bashrc && \
                        echo "conda activate base" >> ~/.bashrc

RUN conda install --yes \
    numpy \
    scipy \
    scikit-image \
    matplotlib \
    s3fs

#RUN pip install awscli --upgrade --user
#RUN pip install --no-cache-dir cupy-cuda101
#RUN pip install cupy
#RUN conda install --yes -c conda-forge cupy
RUN pip install --no-cache-dir cupy-cuda101==9.0.0b2 optuna

ENV HOME /home
ENV PYTHONPATH /home/CEO/python

RUN mkdir /home/CEO
ADD ceo.tar.gz /home/CEO
#RUN cd /home && git clone -b devel_cuda10.1 https://github.com/rconan/CEO.git
RUN cd /home/CEO &&  make all cython
RUN cd /home/CEO/gmtMirrors && ln -s /storage/data03/CEO/gmtMirrors/ASM_fittedKLs_S7OC03945_675kls.ceo ASM_fittedKLs_doubleDiag.ceo

COPY ngsao_windloads.py /usr/local/bin/ngsao_windloads.py

CMD ["nvidia-smi"]
