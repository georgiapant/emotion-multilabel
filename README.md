# Multilable emotion classification

### Dataset
- For the initial experiments the dataset used was the [GoEmotion](https://aclanthology.org/2020.acl-main.372/). All train, validation, test sets can be found in the folder data
- For the generalization tests the dataset used was the [SemEval-ec 2018](https://competitions.codalab.org/competitions/17751) dataset

### Architectures
Several different architectures were used:
- Simple `BERT` with a simple classifier consisting of a fully connected layer followed by a relu activation function, a dropout layer and the final fully connected layer
- `BERT` output stacked with a `biLSTM` layer. There is also the option to use an `LSTM` layer instead
- `BERT - BiLSTM` including additional features extracted from the `NRC lexicon` and the `VAD lexicons`
- `BERT MLM` Simultaneously optimise a loss for both a classification task and a masked language modeling task, the final loss is a linear combination of the two using the `mlm_weight` argument. (Inspiration for this approach was [this](https://arxiv.org/pdf/2109.05782.pdf) paper)

### Additional features
- A weighted loss was implemented to take into account the severe imbalance of the dataset
- A threshold optimisation approach was followed to be able to get the optimal threshold for each class rather than use a specific one for all
- Usage of sparsemax instead of softmax in the self attention architecture of the model (only for `BERT_vad_nrc`)
- Usage of different schedulers including `linear`, `chained` (_step_ and _linear_) and an `adjusted cosine` (only for `BERT_vad_nrc`)
- Option for early stopping based on two metrics, _validation loss_ and _macro F1_

### Results
For a detailed description of the experiments as well as the results please check the report

## To reproduce the results

In the main folder (emotion-multilabel) execute:
```bash
pip install -r requirements.txt
```
Then run the following command:

```bash
python -m emotion_main
```

There are several possible arguments to use when executing the script such as 

* _model_ (default: `BERT`) - other options: `BERT`, `BERT_bilstm`, `BERT_lstm`, `BERT_vad_nrc`, `BERT_MLM`
* _dataset_ (default: `GoEmotions`) - options `GoEmotions`, `EC`
* _max_len_ (default: `126`) 
* _batch_size_ (default: `16`)
* _epochs_ (default: `10`)
* _es_ (default: `f1`) - Metric to check before early stopping, options `f1` and `loss` which is the validation loss
_patience_ (default: `3`) - number of epochs to be patient for early stopping
* _random_seed_ (default: `43`) - to be able to reproduce the results
* _weighted_loss_ (default: `False`) - whether to use or not the weighted loss
* _threshold_opt_ (default: `False`) - whether or not to use the threshold optimization mechanism
* mlm_weight (default: `0.5`)
* _scheduler_ (default: `linear`) - which scheduler to use, options: `linear`, `chained`, `cosine`
* _sparsemax_ (default: `False`) - whether to use sparsemax in the model architecture

For example, to execute the experiment with the BERT vad nrc model with the weighted loss and the chained scheduler 
the below command needs to be executed
```bash
python -m emotion_main --model BERT_vad_nrc --weighted_loss True --scheduler chained
```

### Contact for further details

- Georgia Pantalona (georgia.pant@gmail.com)