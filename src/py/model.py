import pathlib as pl, sqlite3, math, numpy as np, re, os
import keras, tensorflow as tf
import pandas as pd
from keras.layers import Input, Dense, Lambda, Dot, Embedding, Flatten, Multiply, Concatenate, Dropout
from keras.callbacks import EarlyStopping, TensorBoard
from keras.regularizers import l1
from keras.models import Model
import keras.backend as K
import functools, types, atexit
import sklearn as sk
from sklearn.model_selection import GridSearchCV
import importlib
from src.py import model as arch
from tensorboard.plugins.hparams import api as hp
import itertools, random

def isometric_loss(y_true, y_pred):
    
    coef, margin = 10, 0.0
    y_pred = y_pred / tf.norm(y_pred, axis=1, keepdims=True)
    y_true = y_true / tf.norm(y_true, axis=1, keepdims=True)

    inpdif = tf.matmul(y_true,tf.transpose(y_true))
    predif = tf.matmul(y_pred,tf.transpose(y_pred))
    reldif = tf.square(inpdif - predif)
    return coef*reldif

W1 = hp.HParam('num_units', hp.Discrete([250, 500, 1000, 2000]))
COMBINE = hp.HParam('combine', hp.Discrete(['concat','mult']))
ACTIVATION = hp.HParam('activation', hp.Discrete(['relu','tanh','sigmoid']))
DEPTH = hp.HParam('depth', hp.Discrete([4,8]))
DROPOUT = hp.HParam('dropout', hp.Discrete([0.3,0.4]))
EMB_L1 = hp.HParam('emb_l1', hp.Discrete([0.0,1e-3,1e-5]))
LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([0.001,0.0001]))
PREPOST = hp.HParam('prepost', hp.Discrete(['pre','post']))
HP = [W1, PREPOST, COMBINE, DEPTH, DROPOUT, EMB_L1, LEARNING_RATE, ACTIVATION]

def train_test_model(dim,HP):
    # HP = { W1: 100, COMBINE: 'mult', DROPOUT: 0.1, DEPTH: 2, LEARNING_RATE: 0.001 }

    conc = Multiply() if HP[COMBINE]=="mult" else Concatenate()

    inp = Input(shape=(dim,), name='i')
    in1 = Dense(units=HP[W1], activation="relu", name="i1")(inp)
    pid = Input(shape=(1,), name='pid')

    pe = Embedding(name="pid_embedding1", 
        input_dim=500, output_dim=HP[W1], input_length=1, 
        embeddings_regularizer=l1(HP[EMB_L1]))(pid)

    pe = Flatten(name="pid_embedding")(pe)

    at = conc([pe,in1]) if HP[PREPOST]=="pre" else in1
    for i in range(HP[DEPTH]):
        at = Dense(units=HP[W1],activation="relu",name=f"d{i}")(at)
        at = Dropout(HP[DROPOUT])(at)

    at = at if HP[PREPOST]=="pre" else conc([pe,at])
    em = Dense(units=500,activation=HP[ACTIVATION],name=f"embedding")(at)
    at = Dropout(HP[DROPOUT])(em)

    out = Dense(1, activation='sigmoid', name='out')(at)
    
    model = Model(inputs=[inp,pid],outputs=[out,em],name="isomulti")
    model.compile(
        keras.optimizers.Adam(learning_rate=HP[LEARNING_RATE]), 
        loss={"out":"binary_crossentropy","embedding":arch.isometric_loss}, 
        metrics={"out":"accuracy"})

    tenboard = TensorBoard(log_dir='./logs')
    patience = EarlyStopping(patience=3,restore_best_weights=True)
    model.fit(btrn, epochs=100, steps_per_epoch=tstp, batch_size=bdim,
        validation_data=bval, validation_steps=vstp, verbose=1,
        callbacks=[tenboard,patience])
    
    return model.evaluate(bval,steps=vstp,verbose=0)[3]

def param_search():

    os.system('rm -r logs')
    allvals = [h.domain.values for h in HP]
    prod = list(itertools.product(*allvals))
    random.shuffle(prod)

    df = pd.DataFrame(prod,columns=[h.name for h in HP])
    df['accuracy'] = df.index*0.0

    def run(session_number, run_dir, hparams):
        print(f"params are {hparams}")
        with tf.summary.create_file_writer(run_dir).as_default():
            hp.hparams(hparams)  # record the values used in this trial
            accuracy = train_test_model(768,hparams)

            # write to csv and to tensorboard
            df.loc[session_number,'accuracy'] = accuracy
            df.to_csv("logs/hparam_tuning.csv",index=False)
            tf.summary.scalar("accuracy", accuracy, step=2)

    session_num = 0

    for session_num, values in enumerate(prod):
        hparams = {h:v for h,v in zip(HP,values)}
        run_name = "run-%d" % session_num
        print('--- Starting trial: %s' % run_name)
        print({h.name: hparams[h] for h in hparams})
        run(session_num, 'logs/hparam_tuning/' + run_name, hparams)
        session_num += 1