# use the base embeddings and triplet methods to build property specific embeddings 
import pathlib as pl, sqlite3, math, numpy as np, re, os, sys, importlib
import keras, tensorflow as tf
import pandas as pd
from keras.layers import Input, Dense, Lambda, Dot
from keras.models import Model
from src.py import model as arch
from src.py import utilities as util

# TRAIN EMBEDDINGS =======================================================================
src = tf.data.Dataset.load("cache/tfdatasets/activity_embedding")
src = src.map(lambda d: dict(d,**{'pidnum':tf.cast(d['pidnum'],tf.int64)}))

def pid_batches(ds,batchsize = 10000): 
    return ds.group_by_window(
        key_func = lambda x: x['pidnum'],  
        reduce_func = lambda _,batch: batch.batch(batchsize),
        window_size = batchsize)

def munge_trn(row,embedding):
    x = tf.reshape(row[embedding],(bdim,dim))
    p = tf.reshape(row['pidnum'],(bdim,1))
    y = tf.reshape(row['binvalue'],(bdim,1))
    return ((x,p),(y,x))

# SELECT EMBEDDING FOR TRAINING AND BATCH/PROPERTY PARAMETERS =============================
embedding = "emb_chembert"
dim = tf.shape(next(iter(src))[embedding]).numpy()[0]
bdim, maxpidnum = 10000, 500

## Build holdout, validation, and batched training set
chunk = math.ceil(src.cardinality().numpy()*0.1)
trn = src.take(8*chunk)
val = src.skip(8*chunk).take(chunk).cache()
hld = src.skip(9*chunk).cache()

btrn = pid_batches(trn, bdim)
tstp = list(util.collect(btrn,lambda i,x:i))[-1] 
btrn = btrn.unbatch().repeat().batch(bdim).map(lambda x: munge_trn(x,embedding))
btrn = btrn.take(tstp).cache().repeat().prefetch(tf.data.AUTOTUNE)

# BUILD AND SAVE MODEL ====================================================================
model = arch.train_test_model(dim,{},btrn,tstp,val,bdim)
model.save(f'cache/h5model/siamese.h5')

# PLOT EMBEDDINGS =========================================================================
import seaborn as sns
import matplotlib.pyplot as plt

model = keras.models.load_model(f'cache/h5model/siamese.h5', 
    custom_objects={'SiameseLoss': arch.SiameseLoss, "ProjectionConstraint": arch.ProjectionConstraint})
embed = Model(name="isomulti", inputs=model.inputs, outputs=model.get_layer('embedding').output)
pembd = Model(name="pidmulti", inputs=model.inputs[1], outputs=model.get_layer('pid_embedding').output)

def mkplot(data, name):
    plot = sns.clustermap(data, row_cluster=True, col_cluster=True,vmin=0, vmax=1.)
    plt.imshow(data, cmap='hot')
    plt.colorbar()
    plt.savefig(name)

# Create a rank 2 tensor of random float32 values
x = next(iter(val.filter(lambda x: x['pidnum'] == 3).batch(bdim)))

emb = tf.linalg.l2_normalize(x[embedding],axis=1)
sim = tf.matmul(emb,emb,transpose_b=True)
mkplot(emb, "heatmap.png")
mkplot(sim, "heatmap-sim.png")

embx = embed.predict((x[embedding],x["pidnum"])) 
simx = tf.matmul(embx,embx,transpose_b=True)
mkplot(embx, "eheatmap.png")
mkplot(simx, "eheatmap-sim.png")

embp = tf.linalg.l2_normalize(pembd.predict(tf.range(100)))
simp = tf.matmul(embp,embp,transpose_b=True)
mkplot(embp, "pheatmap.png")
mkplot(simp, "pheatmap-sim.png")

topsim = tf.where(tf.logical_and(simx > 0.85 , simx < 0.999))
smix = tf.gather(x['smiles'],topsim[:,0]).numpy()
valx = tf.cast(tf.gather(x['binvalue'],topsim[:,0]),tf.int32).numpy()
smiy = tf.gather(x['smiles'],topsim[:,1]).numpy()
valy = tf.cast(tf.gather(x['binvalue'],topsim[:,1]),tf.int32).numpy()
agg,tot = 0,0
for i in range(smix.shape[0]):
    print(f"{smix[i].decode()} {valx[i]}")
    print(f"{smiy[i].decode()} {valy[i]}")
    print()
    agg = agg + 1 if valx[i] == valy[i] else agg
    tot = tot + 1

print(f"Accuracy: {agg/tot}")