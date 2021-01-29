FROM aftabalam01/pytorch_allennlp:2.0
WORKDIR /job
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
VOLUME ["/job/data", "/job/src", "/job/work", "/job/output"]