import modeling
import tokenization
import attention
import visualization
import tensorflow as tf
import copy
import json
import logging
import math
import os
import torch


def load_tf_weights_in_bert(model, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        print("Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see "
              "https://www.tensorflow.org/install/ for installation instructions.")
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    print("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        print("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split('/')
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m", "global_step"] for n in name):
            print("Skipping {}".format("/".join(name)))
            continue

        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            else:
                l = [m_name]

            if l[0] == 'kernel' or l[0] == 'gamma':
                #                 print('kernel gamma', l, pointer)
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'beta':
                #                 print('beta',l, pointer)
                #                 print('beta로 빠지고', getattr(pointer, 'bias'))
                pointer = getattr(pointer, 'bias')
            #                 print('그 후의', pointer)
            elif l[0] == 'output_bias':
                #                 print('나 실행됨', l, pointer)
                pointer = getattr(getattr(pointer, 'classifier'), 'bias')
            elif l[0] == 'output_weights':
                #                 print('나 실행됨',l, pointer)
                pointer = getattr(getattr(pointer, 'classifier'), 'weight')
            else:
                try:
                    #                     print('try 전의', pointer)
                    pointer = getattr(pointer, l[0])
                #                     print('try 후의', pointer)
                except AttributeError:
                    print("Skipping {}".format("/".join(name)))
                    continue
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        #         print(l)
        #         print(pointer.shape)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        print("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model

bert = "./bert"
bertConfig = modeling.BertConfig.from_json_file('./bert/bert_config.json')
model = modeling.BertForSequenceClassification(bertConfig, 2)
model = load_tf_weights_in_bert(model, bert)
torch.save(model.state_dict(), './pytorch_pt')

