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
import itertools, random, hashlib

W1 = hp.HParam('num_units', hp.Discrete([100, 250, 500, 1000]))
W2 = hp.HParam('num_units_iso', hp.Discrete([100, 250, 500, 1000]))
COMBINE = hp.HParam('combine', hp.Discrete(['concat','mult']))
DEPTH = hp.HParam('depth', hp.Discrete([2,4,6]))
DROPOUT = hp.HParam('dropout', hp.Discrete([0.3,0.5]))
ISOLOSS_C = hp.HParam('isometric_loss_coefficient', hp.Discrete([0.]))
ISOLOSS_MARGIN = hp.HParam('isometric_loss_margin', hp.Discrete([0.]))

PREPOST = hp.HParam('prepost', hp.Discrete(['post',"pre"]))
EMB_L1 = hp.HParam('emb_l1', hp.Discrete([1e-3,1e-6,0.]))
LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([0.01,0.001,0.0001]))
ACTIVATION = hp.HParam('activation', hp.Discrete(['tanh','relu','sigmoid']))
SIAMESE_COEF = hp.HParam('siamese_coef', hp.Discrete([0.1,1.0,10.0]))
SIAMESE_MARG = hp.HParam('siamese_margin', hp.Discrete([0.01,0.05,0.10]))
HPLIST = [W1, W2, PREPOST, COMBINE, DEPTH, DROPOUT, EMB_L1, LEARNING_RATE, ACTIVATION, ISOLOSS_C, ISOLOSS_MARGIN,
      SIAMESE_COEF, SIAMESE_MARG]

class Identity(keras.layers.Layer):
    
    def __init__(self, **kwargs):
        super(Identity, self).__init__(**kwargs)
    
    def call(self, inputs):
        return inputs

def isometric_loss(y_true, y_pred):
        coef, margin = HPLIST[ISOLOSS_C], HPLIST[ISOLOSS_MARGIN]
        y_pred = y_pred / tf.norm(y_pred, axis=1, keepdims=True)
        y_true = y_true / tf.norm(y_true, axis=1, keepdims=True)

        inpdif = tf.matmul(y_true,tf.transpose(y_true))
        predif = tf.matmul(y_pred,tf.transpose(y_pred))
        reldif = tf.reduce_mean(tf.square(inpdif - predif))
        reldif = tf.maximum(reldif-margin,0.0)
        return coef*reldif

class ProjectionConstraint(keras.constraints.Constraint):
    
    def __init__(self, minsum=10, maxsum=100):
        self.minsum, self.maxsum = minsum, maxsum
        self.nonneg = tf.keras.constraints.NonNeg()
        self.minmax = tf.keras.constraints.MinMaxNorm(min_value=minsum**0.5, max_value=maxsum**0.5)
    
    def __call__(self, w):
        print(w)
        w = self.nonneg(w)
        w = self.minmax(w)
        return w

class SiameseLoss(keras.losses.Loss):
    
    def __init__(self, sparse_coefficient=10, sparse_margin=0.05, name="siamese_loss", **kwargs):
        super().__init__(name=name)
        self.sparse_coefficient = sparse_coefficient
        self.sparse_margin = sparse_margin

    def call(self, y_true, y_pred):
        y_true = tf.squeeze(y_true)
        
        predif = 0.5 + tf.matmul(y_pred,tf.transpose(y_pred))/2. # -1 goes to 0, 1 goes to 1
        predif = tf.linalg.set_diag(predif,tf.linalg.diag_part(predif)*0) # remove identity
        
        # DENSE EMBEDDING PENALTY
        sparse_loss = tf.reduce_mean(tf.abs(predif-0.5),axis=1) # penalize each row's high/low similarities
        sparse_loss = tf.maximum(0.,sparse_loss-self.sparse_margin) # allow some high/low sims
        sparse_loss = self.sparse_coefficient * tf.reduce_mean(sparse_loss)
        
        pred1, pred0 = tf.reduce_sum(predif * y_true,axis=1), tf.reduce_sum(predif * (1-y_true),axis=1)
        pred = pred1 / (pred1 + pred0)

        loss = K.mean(K.abs(pred - y_true), axis=-1) + sparse_loss
        loss = tf.maximum(loss,0.0)
        return loss

def train_test_model(dim,HP,btrn,tstp,bval,bdim,vstp):

    HP = {
        arch.W1: 200,
        arch.W2: 200,
        arch.PREPOST: "post", 
        arch.COMBINE: "mult",
        arch.DEPTH: 4, 
        arch.DROPOUT: 0.0, 
        arch.EMB_L1: 1e-9, 
        arch.LEARNING_RATE: 0.001, 
        arch.ACTIVATION: "tanh",
        arch.ISOLOSS_C: 0.1,
        arch.ISOLOSS_MARGIN: 0.1,
        arch.SIAMESE_COEF: 0.1,
        arch.SIAMESE_MARG: 0.05
    }

    conc = Multiply() if HP[COMBINE]=="mult" else Concatenate()

    inp = Input(shape=(dim,), name='i')
    in1 = Dense(units=HP[W1], activation="relu", name="i1")(inp)
    pid = Input(shape=(1,), name='pid')

    pe = Embedding(name="pid_embedding1", 
        input_dim=500, output_dim=HP[W1], input_length=1,
        embeddings_regularizer=keras.regularizers.l1(HP[EMB_L1]),
        embeddings_constraint=ProjectionConstraint(minsum=HP[W1]/10,maxsum=HP[W1]/2.))(pid)

    pe = Flatten(name="pid_embedding")(pe)

    at = conc([pe,in1]) if HP[PREPOST]=="pre" else in1
    for i in range(HP[DEPTH]):
        at = Dense(units=HP[W1],activation="relu",name=f"d{i}")(at)
        at = Dropout(HP[DROPOUT])(at)
        at = keras.layers.BatchNormalization()(at)

    at = at if HP[PREPOST]=="pre" else conc([pe,at])
    
    em = Dense(units=HP[W2],activation=HP[ACTIVATION])(at)
    l1 = Lambda(lambda x: K.l2_normalize(x,axis=1),name=f"embedding")(em)

    model = Model(inputs=[inp,pid],outputs=[l1],name="isomulti")
    sloss = arch.SiameseLoss(HP[arch.SIAMESE_COEF],HP[arch.SIAMESE_MARG])
    model.compile(keras.optimizers.Adam(learning_rate=HP[LEARNING_RATE]), 
        loss=[sloss], 
        # metrics=[siamese_accuracy],
        # run_eagerly=True
        )

    tenboard = TensorBoard(log_dir='./logs')
    patience = EarlyStopping(patience=5,restore_best_weights=True)
    model.fit(btrn, epochs=30, steps_per_epoch=tstp, batch_size=bdim,
        validation_data=bval, validation_steps=vstp, verbose=1,
        callbacks=[tenboard,patience])
    
    return model

def get_md5(file_path):
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()
        
def param_search():

    os.system('rm -r logs')
    importlib.reload(arch)
    allvals = [h.domain.values for h in arch.HPLIST]
    prod = list(itertools.product(*allvals))
    random.shuffle(prod)

    df = pd.DataFrame(prod,columns=[h.name for h in HPLIST])
    df['accuracy'] = df.index*0.0
    df['fileversion'] = get_md5("src/py/model.py")

    def run(session_number, run_dir, hparams):
        print(f"params are {hparams}")
        with tf.summary.create_file_writer(run_dir).as_default():
            hp.hparams(hparams)  # record the values used in this trial
            model = train_test_model(768,hparams,btrn,tstp,bval,bdim,vstp)
            val_loss = model.evaluate(bval,steps=vstp)[0]

            # write to csv and to tensorboard
            df.loc[session_number,'val_loss'] = val_loss
            df.to_csv("logs/hparam_tuning.csv",index=False)
            tf.summary.scalar("val_loss", val_loss, step=2)

    session_num = 0

    for session_num, values in enumerate(prod):
        hparams = {h:v for h,v in zip(HPLIST,values)}
        run_name = "run-%d" % session_num
        print('--- Starting trial: %s' % run_name)
        print({h.name: hparams[h] for h in hparams})
        run(session_num, 'logs/hparam_tuning/' + run_name, hparams)
        session_num += 1
    

   
