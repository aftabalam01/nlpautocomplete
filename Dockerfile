FROM aftabalam01/pytorch_allennlp:2.0
RUN mkdir /job
COPY requirements.txt /job/requirements.txt
RUN pip install -r requirements.txt
WORKDIR /job

VOLUME ["/job/data", "/job/src", "/job/work", "/job/output"]