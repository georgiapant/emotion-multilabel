# project_root_path = r"C:Users\georgiapant\PycharmProjects\GitHub\health_topics"
# project_root_path = "../../"
project_root_path = r"C:\Users\georgiapant\PycharmProjects\GitHub\health_topics"
data_path = "D:REBECCA\Datasets\Search queries topics"


api_url = "http://160.40.51.26:3000" # Browser plugin url
mongodb_parameters = ('localhost', 27017)

MAX_LEN = 126
BATCH_SIZE = 8
# BERT_MODEL = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
# BERT_MODEL = 'bert-base-uncased'
BERT_MODEL = 'bert-base-multilingual-cased'
RANDOM_SEED = 42
EPOCHS = 2
patience = 3


# QIC dataset
# labels = ['treatment plan', 'other', 'disease description', 'Cause analysis', 'Precautions', 'intended effect',
#           'diagnosis', 'medical advice', 'medical fees', 'examination result analysis', 'result description']

# labels = [0,1]

#GoEmotions dataset
# emotions = []

# num_labels = len(labels)