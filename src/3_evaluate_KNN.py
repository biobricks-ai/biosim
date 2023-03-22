import tensorflow as tf, os, pathlib
import pathlib as pl, math, numpy as np, timeit
import re, tensorflow as tf, pandas as pd
import importlib

from src.py import KNN, utilities as util, model as arch

import keras
from keras.layers import Input, Dense, Lambda, Dot
from keras.models import Model

# KNN EVALUATION ===============================================================
# Iterate over each pidnum and evaluate chembert, pubchem2d and supervised embeddings
importlib.reload(arch)

activities = tf.data.Dataset.load("cache/tfdatasets/activity_embedding")
activities = activities.map(lambda d: dict(d,**{'pidnum':tf.cast(d['pidnum'],tf.int32)}))

chunk = math.ceil(activities.cardinality().numpy()/10)
holdout = activities.skip(9*chunk).take(chunk)

# get all unique 'pidnum' values from holdout
pidnums = holdout.map(lambda x: x['pidnum']).unique().cache()
pidnums = [x for x in pidnums.as_numpy_iterator()]

embeddings = ["emb_pubchem2d","emb_chembert","emb_sup"]
embed = arch.load_model("cache/h5model/siamese.h5").embed

os.system("rm -r cache/eval/knn.csv")
with open("cache/eval/knn.csv","a") as f:
    f.write(f"pidnum,emb,prediction,binvalue\n")

for pidnum in pidnums:
    val = holdout.filter(lambda x: x['pidnum'] == pidnum)
    Nval = util.count_and_collect(val)

    # add supervised embedding    
    val = val.repeat().batch(100).take(math.ceil(Nval/100.))
    emb = lambda x: embed((x['emb_chembert'],tf.repeat(tf.constant([pidnum]),100)))
    val = val.map(lambda x: dict(x,**{'emb_sup':emb(x)}))
    val = val.unbatch().take(Nval).cache()
    assert(util.count_and_collect(val) == Nval)
    
    # BASE EMBEDDING ===============================================================
    importlib.reload(KNN)
    for emb in embeddings:
        ds = val.map(lambda x: (x[emb],x['binvalue']))
        rs = KNN.evaluate_knn(ds,k=5).as_numpy_iterator()
        with open("cache/eval/knn.csv","a") as f:
            for x in rs:
                f.write(f"{pidnum},{emb},{x['prediction']},{x['binvalue']}\n")