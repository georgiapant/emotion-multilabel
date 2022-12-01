# Multilable emotion classification

### Dataset
- For the initial experiments the dataset used was the [GoEmotion](https://aclanthology.org/2020.acl-main.372/). All train, validation, test sets can be found in the folder data
- For the generalization tests the dataset used was the [SemEval-ec 2018](https://competitions.codalab.org/competitions/17751) dataset

### Architectures
Several different architectures were used:
- Simple BERT with a simple classifier consisting of a fully connected layer followed by a relu activation function, a dropout layer and the final fully connected layer
- BERT output stacked with a biLSTM layer. There is also the option to use an LSTM layer instead
- BERT - BiLSTM including additional features extracted from the NRC lexicon and the VAD lexicons
- BERT MLM ...

### Additional features
- A weighted loss was implemented to take into account the severe imbalance of the dataset
- A threshold optimisation approach was followed to be able to get the optimal threshold for each class rather than use a specific one for all
- Usage of sparsemax instead of softmax in the self attention architecture of the model (only for BERT_vad_nrc)
- Usage of different schedulers including linear, chained (step and linear) and an adjusted cosine (only for BERT_vad_nrc)
- Option for early stopping based on two metrics, validation loss and macro F1

### Results
For a detailed description of the experiments as well as the results please check the report

## To reproduce the results

In the main folder (emotion-multilabel) execute:
```bash
pip install -r requirements.txt
```
Then run the following command:

```python
python -m emotion_main
```

There are several possible arguments to use when executing the script such as 

* model (default: `BERT`) - other options: `BERT`, `BERT_bilstm`, `BERT_lstm`, `BERT_vad_nrc`, `BERT_MLM`
* dataset (default: `GoEmotions`) - options `GoEmotions`, `EC`
* max_len (default: `126`) 
* batch_size (default: `16`)
* epochs (default: `10`)
* es (default: `f1`) - Metric to check before early stopping, options `f1` and `loss` which is the validation loss
* patience (default: `3`) - number of epochs to be patient for early stopping
* random_seed (default: `43`) - to be able to reproduce the results
* weighted_loss (default: `False`) - whether to use or not the weighted loss
* threshold_opt (default: `False`) - whether or not to use the threshold optimization mechanism
* mlm_weight
* scheduler (default: `linear`) - which scheduler to use, options: `linear`, `chained`, `cosine`
* sparsemax (default: `False`) - whether to use sparsemax in the model architecture

For example, to execute the experiment with the BERT vad nrc model with the weighted loss and the chained scheduler 
the below command needs to be executed
```python
python -m emotion_main --model BERT_vad_nrc --weighted_loss True --scheduler chained
```