FROM tensorflow/tensorflow:2.1.0-py3-jupyter

WORKDIR /notebooks
ADD requirements.txt /notebooks/requirements.txt

RUN python -m pip install -r requirements.txt
COPY . /notebooks

CMD jupyter notebook --ip=0.0.0.0 --allow-root