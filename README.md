# IJAIED_code_for_MWAN

This repository contains source code(Pytorch) to reproduce the results presented in the paper "Automatic short answer grading", IJAIED 2020

## Note
- The source code for CMAN is in [./IJAIED/model_zoo/model.py]
- The source code for CMAN++ is in [./IJAIED/model_zoo/new_model.py]

## Prerequisites
#### 1 - Install Requirements
```
pip install -r requirements.txt
```

## Project structure
* 'IJAIE/'
    * 'logs/' -- this folder will generate the training logs automatically if you set "-log" with a log name.
    * 'model_zoo/' -- this folder contains the source code for our proposed models.
    * 'saved_model/' -- this folder will save the best performance model checkpoint if you set "-save_model" with a model name.
    * 'semeval/' -- this folder contains the training data (vocabulary, sentence after tokenization and padding).
    * 'new_model_*way_*.py/' -- .py file for training the proposed model.
    * 'semeval_text/' -- this folder contains the unprocessed text data.


## Training
train 2way task on unseen answer dataset with extended model
```
python ./IJAIED/new_model_2way.py -log ./logs/2way_model_extension -save_model ./saved_model/2way_model_extension -epoch 200
```

train 2way task on unseen question & domain dataset with extended model
```
python ./IJAIED/new_model_2way_que.py -log ./logs/2way_model_extension_que -save_model ./saved_model/2way_model_extension_que -epoch 200
```

train 3way task on unseen answer dataset with extended model
```
python ./IJAIED/new_model_3way.py -log ./logs/3way_model_extension -save_model ./saved_model/3way_model_extension -epoch 200
```

train 2way task on unseen answer dataset with basic model
```
python ./IJAIED/train_semeval_ans.py -log ./logs/2way_model -save_model ./saved_model/2way_model -epoch 200
```
