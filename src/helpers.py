
import datetime
import numpy as np
import torch
import random
import json
import unicodedata
import six
import sys
project_root_path = "C:/Users/georgiapant/PycharmProjects/GitHub/rebecca"
sys.path.append(project_root_path)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round(elapsed))

    # Format as hh:mm:ss.microsecs
    # return str(datetime.timedelta(seconds=elapsed_rounded))
    return str(datetime.timedelta(seconds=elapsed_rounded, microseconds=elapsed_rounded))


def read_file(filename):
    f = open(filename, 'r')
    lines = f.readlines()
    label = []
    text = []
    for l in lines:
        label.append(int(l[0]))
        text.append(l[2:])

    label = np.array(label)
    return text, label


def set_seed(seed_value=42):
    """
    Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def probs_to_labels_multilabel(probs):
    y_pred = []

    for i in range(len(probs)):
        pred = np.where(probs[i] == np.amax(probs[i]))[0][0]
        y_pred.append(pred)

    with open(project_root_path+ "/data/interim/emotion_mappings.json") as file:
        file_data = file.read()
        mapping = json.loads(file_data)

    rev_mapping = {v: k for k, v in mapping.items()}
    predicted_labels = [rev_mapping.get(item, item) for item in y_pred]

    return predicted_labels


def convert_to_unicode(text):
    """
    Converts `text` to Unicode (if it's not already), assuming UTF-8 input.
    """

    # six_ensure_text is copied from https://github.com/benjaminp/six
    def six_ensure_text(s, encoding='utf-8', errors='strict'):
        if isinstance(s, six.binary_type):
            return s.decode(encoding, errors)
        elif isinstance(s, six.text_type):
            return s
        else:
            raise TypeError("not expecting type '%s'" % type(s))

    return six_ensure_text(text, encoding="utf-8", errors="ignore")


def run_strip_accents(text):
    """
    Strips accents from a piece of text.
    """
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat == "Mn":
            continue
        output.append(char)
    return "".join(output)


