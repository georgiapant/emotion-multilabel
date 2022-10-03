import pandas as pd
from collections import Counter
import json
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.utils import compute_class_weight


# from config import data_path, project_root_path


class CreateDataset:
    def __init__(self, project_root_path):
        self.project_root_path = project_root_path

    def calculating_class_weights(self, y_true, class_names):
        number_dim = np.shape(y_true)[1]
        weights = np.empty([number_dim, 2])
        for i in range(len(class_names)):
            weights[i] = compute_class_weight('balanced', classes=[0, 1], y=y_true[class_names[i]])
        return weights

    def pos_weights(self, y_true, class_names):
        # print(y_true)
        weights = []
        for i in range(len(class_names)):
            num_pos = np.sum(y_true[class_names[i]])
            num_neg = len(y_true[class_names[i]]) - num_pos
            weights.append(num_neg/num_pos)
        return weights

    def goemotions(self, drop_neutral=False, with_weights=False):

        stats = pd.DataFrame()
        weights = None

        train = pd.read_csv(self.project_root_path + "/data/train.tsv",
                            encoding="utf8", low_memory=False, sep='\t', names=['Text', 'sentiment_id', 'id'])

        validation = pd.read_csv(
            self.project_root_path + "/data/dev.tsv",
            encoding="utf8", low_memory=False, sep='\t', names=['Text', 'sentiment_id', 'id'])

        test = pd.read_csv(
            self.project_root_path + "/data/test.tsv",
            encoding="utf8", low_memory=False, sep='\t', names=['Text', 'sentiment_id', 'id'])

        # Loading emotion labels for GoEmotions taxonomy
        with open(self.project_root_path + "/data/emotions.txt", "r") as file:
            GE_taxonomy = file.read().split("\n")

        size_train = train.shape[0]
        size_val = validation.shape[0]
        size_test = test.shape[0]

        df_all = pd.concat([train, validation, test], axis=0).reset_index(drop=True).drop(['id'], axis=1)
        df_all['sentiment_id'] = df_all['sentiment_id'].apply(lambda x: x.split(','))

        def idx2class(idx_list):
            arr = []
            for i in idx_list:
                arr.append(GE_taxonomy[int(i)])
            # print(arr)
            return arr

        df_all['sentiment'] = df_all['sentiment_id'].apply(idx2class)

        # OneHot encoding for multi-label classification
        for emo in GE_taxonomy:
            df_all[emo] = np.zeros((len(df_all), 1))
            df_all[emo] = df_all['sentiment'].apply(lambda x: 1 if emo in x else 0)

        df_all = df_all.drop(['sentiment_id', 'sentiment'], axis=1)

        X_train = df_all.iloc[:size_train, :]['Text']
        X_val = df_all.iloc[size_train:size_train + size_val, :]['Text']
        X_test = df_all.iloc[size_train + size_val:size_train + size_val + size_test, :]['Text']

        y_train = df_all.iloc[:size_train, :].drop(columns=['Text'])
        y_val = df_all.iloc[size_train:size_train + size_val, :].drop(columns=['Text'])
        y_test = df_all.iloc[size_train + size_val:size_train + size_val + size_test, :].drop(columns=['Text'])

        if drop_neutral:
            X_train, y_train, X_val, y_val, X_test, y_test = self.dropNeutral(X_train, y_train, X_val, y_val, X_test,
                                                                              y_test)

        stats['Train'] = y_train.sum(axis=0)
        stats['Val'] = y_val.sum(axis=0)
        stats['Test'] = y_test.sum(axis=0)
        labels = y_train.columns
        print(stats)

        if with_weights:
            weights = self.calculating_class_weights(y_train, labels)
            # weights = self.pos_weights(y_train, labels)
            print(f'The weights of the classes are:\n {weights}')

        return X_train, y_train, X_val, y_val, X_test, y_test, labels, weights

    def dropNeutral(self, X_train, y_train, X_val, y_val, X_test, y_test):
        temp = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
        temp.drop(temp[temp['neutral'] == 1].index, inplace=True)
        temp = temp.drop(columns=['neutral'])
        X_train = temp['Text']
        y_train = temp.drop(['Text'], axis=1)

        temp = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)
        temp.drop(temp[temp['neutral'] == 1].index, inplace=True)
        temp = temp.drop(columns=['neutral'])
        X_test = temp['Text']
        y_test = temp.drop(['Text'], axis=1)

        temp = pd.concat([X_val, y_val], axis=1).reset_index(drop=True)
        temp.drop(temp[temp['neutral'] == 1].index, inplace=True)
        temp = temp.drop(columns=['neutral'])
        X_val = temp['Text']
        y_val = temp.drop(['Text'], axis=1)

        return X_train, y_train, X_val, y_val, X_test, y_test

    def ec(self, with_weights=False):
        data_ec_val = pd.read_csv(self.project_root_path + "/data/ec/2018-E-c-En-dev.txt",
            sep='	', encoding="utf-8", header=0)
        data_ec_test = pd.read_csv(self.project_root_path + "/data/ec/2018-E-c-En-test-gold.txt",
             sep='	', encoding="utf-8", header=0)
        data_ec_train = pd.read_csv(self.project_root_path + "/data/ec/2018-E-c-En-train.txt",
            sep='	', encoding="utf-8", header=0)

        stats = pd.DataFrame()
        weights = None

        data_ec_val.rename(columns={"Tweet": "Text"}, inplace=True)
        data_ec_test.rename(columns={"Tweet": "Text"}, inplace=True)
        data_ec_train.rename(columns={"Tweet": "Text"}, inplace=True)

        X_train = data_ec_train['Text'].values
        y_train = data_ec_train.drop(columns=['Text', 'ID'])

        X_val = data_ec_val['Text'].values
        y_val = data_ec_val.drop(columns=['Text', 'ID'])

        X_test = data_ec_test['Text'].values
        y_test = data_ec_test.drop(columns=['Text', 'ID'])

        stats = pd.DataFrame()
        stats['Train'] = y_train.sum(axis=0)
        stats['Val'] = y_val.sum(axis=0)
        stats['Test'] = y_test.sum(axis=0)
        labels = y_train.columns
        print(stats)

        if with_weights:
            weights = self.calculating_class_weights(y_train, labels)
            # weights = self.pos_weights(y_train, labels)
            print(f'The weights of the classes are:\n {weights}')

        return X_train, y_train, X_val, y_val, X_test, y_test, labels, weights
