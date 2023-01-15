# use the base embeddings and triplet methods to build property specific embeddings 
import pathlib as pl, sqlite3, math, numpy as np, re, os, sys, importlib
import keras, tensorflow as tf
import pandas as pd
from keras.layers import Input, Dense, Lambda, Dot
from keras.models import Model
from src.py import model as arch

# TRAIN EMBEDDINGS =======================================================================
# def train_transfer_ff(trn,val):

src = tf.data.Dataset.load("cache/tfdatasets/activity_embedding").shuffle(100000)

chunk = math.ceil(src.cardinality().numpy()*0.1)
trn = src.take(8*chunk).cache()
val = src.skip(8*chunk).take(chunk).cache()
hld = src.skip(9*chunk).cache()

xtrn = next(iter(trn.batch(100)))

embedding = "emb_chembert"
# embedding = "emb_pubchem2d" 
dim = tf.shape(next(iter(trn))[embedding]).numpy()[0]
maxpidnum = 500 #trn.reduce(0., lambda x,y: tf.maximum(x,y['pidnum'])).numpy()
bdim = 10000

def munge_trn(row):
    x = tf.reshape(row[embedding],(bdim,dim))
    p = tf.reshape(row['pidnum'],(bdim,1))
    y = tf.reshape(row['binvalue'],(bdim,1))
    return ((x,p),(y,x))

tstp = math.floor(trn.cardinality().numpy()/bdim)
btrn = trn.batch(bdim).take(tstp).map(munge_trn).cache().repeat().prefetch(tf.data.AUTOTUNE)
vstp = math.floor(val.cardinality().numpy()/bdim)
bval = val.batch(bdim).take(vstp).map(munge_trn).cache().prefetch(tf.data.AUTOTUNE)

def fit(natoms=20):
    importlib.reload(arch)
    model,params = arch.transfer_ff(dim,3)
    tenboard = tf.keras.callbacks.TensorBoard(log_dir='./logs')
    patience = keras.callbacks.EarlyStopping(patience=50,restore_best_weights=True,monitor="val_accuracy")
    model.fit(btrn,epochs=1000,steps_per_epoch=tstp,batch_size=bdim,
        validation_data=bval, validation_steps=vstp, verbose=1,
        callbacks=[tenboard,patience])
    return model

model = fit(natoms=100)
model.save(f'cache/tmp/model.h5')
# load model
model = keras.models.load_model(f'cache/tmp/model.h5', custom_objects={'isometric_loss': arch.isometric_loss})
embed = Model(name="isomulti", inputs=model.inputs, outputs=model.get_layer('embedding').output)
pembd = Model(name="pidmulti", inputs=model.inputs[1], outputs=model.get_layer('pid_embedding').output)

it = iter(btrn.rebatch(100))
x = next(it)
emb = embed.predict(x[0])
emb = pembd.predict(tf.constant(list(range(1,200))))

import seaborn as sns

import matplotlib.pyplot as plt

# Create a rank 2 tensor of random float32 values
data = emb

plot = sns.clustermap(data, row_cluster=True)

# Create a heatmap from the tensor
plt.imshow(data, cmap='hot')
plt.colorbar()
plt.savefig("heatmap.png")

inpdif = tf.matmul(x[0][0],x[0][0],transpose_b=True)
# output should have aid, pid, embedding_type, smiles, array, value
outpath = pl.Path(f'cache/tmp/embeddings')
src.map(lambda x : 
    x[model.name])

edf = hld.map(lambda i,p,e,smi,arr,v: (i,p,"isodense",smi,tf.squeeze(embed(arr)),v)).cache()

def embed(arr):
    return model.predict(arr)

