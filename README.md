# NLP DNN baselines
DNN baselines, implemented by Tensorflow 2.0

## contents
### preprocess
- tf.data.Dataset: preprocess/tfdata.py
    - wakati by Janome: preprocess/custom_tokenizer.py
### architectures
- LSTM: lstm.py
- GRU: gru.py

## example usage
```
$ docker-compose build
$ docker-compose up
$ docker-compose run tf python nlp-dnn-baselines/lstm.py
```