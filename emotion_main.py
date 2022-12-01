import argparse

from src.models.BERT.BERT_train_eval import BertSimple
from src.models.BERT_biLSTM.BERT_BiLSTM_train_eval import BertBilstm
from src.models.NRC_VAD.BERT_VAD_NRC_train_eval import BertVadNrc
from src.models.BERT_MLM.BERT_MLM_train_eval import BertMLM

MODELS = {"BERT": BertSimple, "BERT_bilstm": BertBilstm, "BERT_lstm": BertBilstm,
          "BERT_vad_nrc": BertVadNrc,"BERT_MLM": BertMLM}


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='BERT')
parser.add_argument('--dataset', type=str, default='goemotions')
parser.add_argument('--max_len', type=int, default=126)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--patience', type=int, default=3)
parser.add_argument('--random_seed', type=int, default=42)
parser.add_argument('--weighted_loss', type=bool, default=False)
parser.add_argument('--threshold_opt', type=bool, default=False)
parser.add_argument('--mlm_weight', type=float, default=0.5)
parser.add_argument('--es', type=str, default='f1')
parser.add_argument('--scheduler', type=str, default='linear')
parser.add_argument('--sparsemax', type=bool, default=False)


BERT_MODEL = 'bert-base-uncased'
project_root_path = "./"

args = parser.parse_args()

model_cls = MODELS[args.model]

if args.model == 'BERT_bilstm':
    bidirectional = True
else:
    bidirectional = False

model = model_cls(args.dataset, args.drop_neutral, args.weighted_loss, args.threshold_opt, args.batch_size,
                  args.max_len, args.epochs, args.patience, BERT_MODEL, bidirectional, args.mlm_weight,
                  args.random_seed, project_root_path, args.es, args.scheduler, args.sparsemax)
model.main()
