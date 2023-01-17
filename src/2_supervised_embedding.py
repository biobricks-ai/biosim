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
def munge_src(row):
    row['pidnum'] = tf.cast(row['pidnum'],tf.int64)
    return row
src = src.map(munge_src)

def pid_batches(ds,batchsize = 10000): 
    return ds.group_by_window(
        key_func = lambda x: x['pidnum'],  
        reduce_func = lambda _,batch: batch.batch(batchsize),
        window_size = batchsize)

embedding="emb_chembert"
bdim = 10000
dim = tf.shape(next(iter(src))[embedding]).numpy()[0]
maxpidnum = 500 #trn.reduce(0., lambda x,y: tf.maximum(x,y['pidnum'])).numpy()

def munge_trn(row):
    x = tf.reshape(row[embedding],(bdim,dim))
    p = tf.reshape(row['pidnum'],(bdim,1))
    y = tf.reshape(row['binvalue'],(bdim,1))
    return ((x,p),(y,x))

chunk = math.ceil(src.cardinality().numpy()*0.1)

trn  = pid_batches(src.take(8*chunk))
btrn = trn.unbatch().repeat().batch(bdim).map(munge_trn).take(325).cache().repeat().prefetch(tf.data.AUTOTUNE)

# tstp = [i for i,x in enumerate(btrn)][-1]
x = next(iter(btrn))
tstp, minbatch = 0, 10000000
for i,x in enumerate(btrn):
    tstp, minbatch = i, min(minbatch,tf.shape(x[0][1])[0].numpy())
    if i > 350: 
        break
    print(minbatch)
    print(i)

val  = src.skip(8*chunk).take(chunk).cache()
vstp = math.floor(val.cardinality().numpy()/bdim)
bval = val.batch(bdim).take(vstp).map(munge_trn).cache().prefetch(tf.data.AUTOTUNE)

hld  = src.skip(9*chunk).cache()



model = arch.train_test_model(dim,{},btrn,tstp,bval,bdim,vstp)
model.save(f'cache/siamese-model.h5')
# load model with the SiameseLoss custom loss class
model = keras.models.load_model(f'cache/siamese-model.h5', 
    custom_objects={'SiameseLoss': arch.SiameseLoss, "ProjectionConstraint": arch.ProjectionConstraint})

embed = Model(name="isomulti", inputs=model.inputs, outputs=model.get_layer('embedding').output)
pembd = Model(name="pidmulti", inputs=model.inputs[1], outputs=model.get_layer('pid_embedding').output)

it = iter(btrn.rebatch(1000))
x = next(it)
emb = embed.predict(x[0])
emb = pembd.predict(tf.range(500))

import seaborn as sns

import matplotlib.pyplot as plt

# Create a rank 2 tensor of random float32 values
data = emb

plot = sns.clustermap(data, row_cluster=True, col_cluster=True)

# Create a heatmap from the tensor
plt.imshow(data, cmap='hot')
plt.colorbar()
plt.savefig("heatmap-emb.png")

inpdif = tf.matmul(x[0][0],x[0][0],transpose_b=True)
# output should have aid, pid, embedding_type, smiles, array, value
outpath = pl.Path(f'cache/tmp/embeddings')
src.map(lambda x : 
    x[model.name])

edf = hld.map(lambda i,p,e,smi,arr,v: (i,p,"isodense",smi,tf.squeeze(embed(arr)),v)).cache()

def embed(arr):
    return model.predict(arr)

