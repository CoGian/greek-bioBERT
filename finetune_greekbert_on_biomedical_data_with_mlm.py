# -*- coding: utf-8 -*-
"""Finetune greekBERT on biomedical data with MLM.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1VIMfPxYIcvu5UqYWHga87UbErkcOkp43
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.optimizers import Adam
import transformers
from transformers import TFAutoModelWithLMHead, AutoTokenizer
import logging
import json
# no extensive logging 
logging.getLogger().setLevel(logging.NOTSET)

AUTO = tf.data.experimental.AUTOTUNE

with open("preprocess/sentences_dataset.json", "r" , encoding="utf-8") as fin:
  sentences = json.load(fin)

MAX_LEN = 128
BATCH_SIZE = 8 # per TPU core
TOTAL_STEPS = (len(sentences) // (BATCH_SIZE * 8) ) * 4 # thats approx 4 epochs
EVALUATE_EVERY = (len(sentences) // (BATCH_SIZE * 8) ) // 2
LR =  1e-5

PRETRAINED_MODEL = 'nlpaueb/bert-base-greek-uncased-v1'

def connect_to_TPU():
    """Detect hardware, return appropriate distribution strategy"""
    try:
        # TPU detection. No parameters necessary if TPU_NAME environment variable is
        # set: this is always the case on Kaggle.
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Running on TPU ', tpu.master())
    except ValueError:
        tpu = None

    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
    else:
        # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
        strategy = tf.distribute.get_strategy()

    global_batch_size = BATCH_SIZE * strategy.num_replicas_in_sync

    return tpu, strategy, global_batch_size


tpu, strategy, global_batch_size = connect_to_TPU()
print("REPLICAS: ", strategy.num_replicas_in_sync)


def regular_encode(texts, tokenizer, maxlen=512):
    enc_di = tokenizer.batch_encode_plus(
        texts,
        return_attention_mask=False,
        return_token_type_ids=False,
        pad_to_max_length=True,
        max_length=maxlen,
        truncation=True
    )

    return np.array(enc_di['input_ids'])


tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
X_train = regular_encode(sentences, tokenizer, maxlen=MAX_LEN)

def prepare_mlm_input_and_labels(X):
    # 15% BERT masking
    inp_mask = np.random.rand(*X.shape)<0.15 
    # do not mask special tokens
    inp_mask[X<=103] = False
    # set targets to -1 by default, it means ignore
    labels =  -1 * np.ones(X.shape, dtype=int)
    # set labels for masked tokens
    labels[inp_mask] = X[inp_mask]
    
    # prepare input
    X_mlm = np.copy(X)
    # set input to [MASK] which is the last token for the 90% of tokens
    # this means leaving 10% unchanged
    inp_mask_2mask = inp_mask  & (np.random.rand(*X.shape)<0.90)
    X_mlm[inp_mask_2mask] = 103  # mask token is the last in the dict

    # set 10% to a random token
    inp_mask_2random = inp_mask_2mask  & (np.random.rand(*X.shape) < 1/9)
    X_mlm[inp_mask_2random] = np.random.randint(3, 250001, inp_mask_2random.sum())
    
    return X_mlm, labels

# masks and labels
X_train_mlm, y_train_mlm = prepare_mlm_input_and_labels(X_train)

def create_dist_dataset(X, y=None, training=False):
    dataset = tf.data.Dataset.from_tensor_slices(X)

    ### Add y if present ###
    if y is not None:
        dataset_y = tf.data.Dataset.from_tensor_slices(y)
        dataset = tf.data.Dataset.zip((dataset, dataset_y))
        
    ### Repeat if training ###
    if training:
        dataset = dataset.shuffle(len(X)).repeat()

    dataset = dataset.batch(global_batch_size).prefetch(AUTO)

    ### make it distributed  ###
    dist_dataset = strategy.experimental_distribute_dataset(dataset)

    return dist_dataset
    
    
train_dist_dataset = create_dist_dataset(X_train_mlm, y_train_mlm, True)

def create_mlm_model_and_optimizer():
    with strategy.scope():
        model = TFAutoModelWithLMHead.from_pretrained(PRETRAINED_MODEL)
        optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
    return model, optimizer


mlm_model, optimizer = create_mlm_model_and_optimizer()
mlm_model.summary()

def define_mlm_loss_and_metrics():
    with strategy.scope():
        mlm_loss_object = masked_sparse_categorical_crossentropy

        def compute_mlm_loss(labels, predictions):
            per_example_loss = mlm_loss_object(labels, predictions)
            loss = tf.nn.compute_average_loss(
                per_example_loss, global_batch_size = global_batch_size)
            return loss

        train_mlm_loss_metric = tf.keras.metrics.Mean()
        
    return compute_mlm_loss, train_mlm_loss_metric


def masked_sparse_categorical_crossentropy(y_true, y_pred):
    y_true_masked = tf.boolean_mask(y_true, tf.not_equal(y_true, -1))
    y_pred_masked = tf.boolean_mask(y_pred, tf.not_equal(y_true, -1))
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true_masked,
                                                          y_pred_masked,
                                                          from_logits=True)
    return loss

            
            
def train_mlm(train_dist_dataset, total_steps=2000, evaluate_every=200):
    step = 0
    ### Training lopp ###
    for tensor in train_dist_dataset:
        distributed_mlm_train_step(tensor) 
        step+=1

        if (step % evaluate_every == 0):   
            ### Print train metrics ###  
            train_metric = train_mlm_loss_metric.result().numpy()
            print("Step %d, train loss: %.2f" % (step, train_metric))     

            ### Reset  metrics ###
            train_mlm_loss_metric.reset_states()
            
        if step  == total_steps:
            break


@tf.function
def distributed_mlm_train_step(data):
    strategy.run(mlm_train_step, args=(data,))


@tf.function
def mlm_train_step(inputs):
    features, labels = inputs

    with tf.GradientTape() as tape:
        predictions = mlm_model(features, training=True)[0]
        loss = compute_mlm_loss(labels, predictions)

    gradients = tape.gradient(loss, mlm_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, mlm_model.trainable_variables))

    train_mlm_loss_metric.update_state(loss)
    

compute_mlm_loss, train_mlm_loss_metric = define_mlm_loss_and_metrics()

train_mlm(train_dist_dataset, TOTAL_STEPS, EVALUATE_EVERY)

mlm_model.save_pretrained('greekBERT')
tokenizer.save_pretrained("greekBERT")