# Comparison of state-of-the-art neural network architectures in the task of predicting clinical events over medical temporal sequences
Code used for paper Comparison of state-of-the-art neural network architectures in the task of predicting clinical events over medical temporal sequences

## Project setup
We recommend using conda with python 3.7 and tensorflow 1.5:
```
conda create --name py37_tensorflow python=3.7
conda install numpy==1.19
conda install -c conda-forge tensorflow-gpu=1.15
```

### Execution example (from project main dir):

#### Training (for ICD-9 input codes data)
`python3.7 models/lstm/LSTM.py "data/mimic_90-10_855" compiled_models/lstm-model --hiddenDimSize=[1084]`

#### Testing (for ICD-9 input codes data)
`python3.7 models/lstm/LSTM-test.py "data/mimic_90-10_855" compiled_models/lstm-model.50/`