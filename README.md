# Multilable emotion classification

### Dataset
- For the initial experiments the dataset used was the GoEmotion. All train, validation, test sets can be found in the folder data
- For the generalization tests the dataset used was the ...

### Architectures
Several different architectures were used:
- Simple BERT with a simple classifier consisting of a fully connected layer followed by a relu activation function, a dropout layer and the final fully connected layer
- BERT output stacked with a biLSTM layer. There is also the option to use an LSTM layer instead
- BERT - BiLSTM including additional features extracted from the NRC lexicon and the VAD lexicons

### Additional features
- A weighted loss was implemented to take into account the severe imbalance of the dataset
- A threshold optimisation approach was followed to be able to get the optimal threshold for each class rather than use a specific one for all

The additional features were used in all different model architectures

### Results
...

### Outcomes

## To reproduce the results
...
