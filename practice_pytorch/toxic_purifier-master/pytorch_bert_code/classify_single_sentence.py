from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys
from datetime import datetime

import numpy as np
import math
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

from file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from modeling import BertForSequenceClassification, BertConfig, BertModel
from tokenization import BertTokenizer
from optimization import BertAdam, WarmupLinearSchedule

from attention_map import AttentionMapData, show

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class PuriProcessor(DataProcessor):
    '''Processor for classify toxic expression'''

    def get_train_examples(self, data_dir):
        return self.create_examples(self.read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self.create_examples(self.read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        return ["0", "1"]

    def create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def create_single_example(self, text):
        """Creates single exmaples for predicting a single sentence"""
        guid = 0
        text_a = text
        text_b = None
        label = None
        example = InputExample(guid=guid,
                               text_a=text_a,
                               text_b=text_b,
                               label=None)

        return example


def convert_single_example_to_feature(example, max_seq_length, tokenizer, output_mode):
    '''Convert single InputExample to InputFeature for predict'''
    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None

    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[:(max_seq_length - 2)]

    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids = [0] * len(tokens)

    # convert tokens to vocab index
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    # check each element's length
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    # we only want to predict task
    label_id = None

    # convert example to feature
    feature = InputFeatures(input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids,
                            label_id=label_id)

    return feature


def predict_single_sentence(model_wieght_path, text):
    # initialize basic options
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    output_mode = 'classification'
    processor = PuriProcessor()
    label_list = processor.get_labels()
    num_labels = len(label_list)

    bert_model = 'bert-base-multilingual-uncased'
    tokenizer = BertTokenizer.from_pretrained(bert_model,
                                              do_lower_case=True)

    # convert text to feature
    example = processor.create_single_example(text)
    feature = convert_single_example_to_feature(example, 128, tokenizer, output_mode)

    # create_model
    model = BertForSequenceClassification.from_pretrained(model_wieght_path,
                                                          num_labels=num_labels)

    model.eval()
    preds = []

    input_ids = torch.tensor(feature.input_ids, dtype=torch.long).unsqueeze(0)
    input_mask = torch.tensor(feature.input_mask, dtype=torch.long).unsqueeze(0)
    segment_ids = torch.tensor(feature.segment_ids, dtype=torch.long).unsqueeze(0)
    #     label_ids  = feature.label_ids

    with torch.no_grad():
        logits = model(input_ids, segment_ids, input_mask, labels=None)
    if len(preds) == 0:
        preds.append(logits.detach().cpu().numpy())
    else:
        preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)

    preds = preds[0]
    result = np.argmax(preds, axis=1)

    return preds, result

model_weight_path = './pytorch_model/'
text = '한 문장 예측 좀 도와주세요.'

predict_single_sentence(model_weight_path, text)

text3 = '사장님 여기 우동사리 하나 더 주세요!!'
a = datetime.now()
predict_single_sentence(model_weight_path, text3)
b= datetime.now()
print(b-a)

