import tensorflow as tf, os, pathlib
import pathlib as pl, math, numpy as np, timeit
import re, tensorflow as tf, pandas as pd

def knn(path):
    path = "cache/chemprop_tfds/embedding=bert/pid=84253fff-8d28-4db4-b641-2938dd29db48"
    path = "cache/chemprop_tfds/embedding=isodense/pid=84253fff-8d28-4db4-b641-2938dd29db48"
    
    d1 = tf.data.Dataset.load(path) # aid, pid, emb, smiles, array, value
    d1 = d1.map(lambda a,p,e,s,arr,v: (tf.squeeze(arr),v)).cache()
    cmp = d1.batch(10000)

    batch = next(iter(cmp))
    x = next(iter(d1))
    ai = x[0] 

    @tf.function
    def knn(ai,vi):
        
        def knn_map(aj,vj):
            bsim = tf.tensordot(ai,tf.transpose(aj),axes=1)
            maxsim,maxidx = tf.nn.top_k(bsim, k=5,sorted=False)
            maxval = tf.gather(vj, indices=maxidx)
            return (maxsim, maxval)

        def knn_red(agg, row):
            sims, vals = row
            ind = tf.where(tf.logical_and(tf.less(sims,0.999),tf.greater(sims,0.9)))
            sims = tf.squeeze(tf.gather(sims, ind))
            sims = tf.concat([sims,agg[0]],axis=0)
            vals = tf.squeeze(tf.gather(vals, ind))
            vals = tf.concat([vals,agg[1]],axis=0)
            return (sims,vals,agg[2]+tf.shape(vals)[0])

        init = (tf.constant([],dtype=tf.float32),tf.constant([],dtype=tf.int32),0)
        return cmp.map(knn_map).reduce(init,knn_red)

    @tf.function
    def vote(sims,vals):
        weights = tf.tensordot(sims,tf.cast(vals,dtype=tf.float32),axes=1)
        return tf.reduce_sum(weights)/tf.reduce_sum(sims)
    d2 = d1.map(lambda ai,vi: (knn(ai,vi),vi)).filter(lambda x,y: tf.greater(x[2],0))
    d2 = d2.map(lambda x,y: (vote(x[0],x[1]),y))
    
    it = iter(d3)
    next(it)

    d3 = d2.map(lambda p,v: (tf.cast(p > 0.5, dtype=tf.int32),v))
    d3.reduce(tf.zeros([2,2]), lambda agg, row: tf.tensor_scatter_nd_add(agg, [[row[0],row[1]]], [1]))
    x = next(iter(d3))


def build_property_evaluation_tfds(path):

    d1 = tf.data.Dataset.load(path)
    d2 = d1.map(lambda a,p,e,s,arr,v: (s,tf.strings.to_hash_bucket_fast(s,100e6),v,arr)).batch(10000)
    d2 = d2.map(lambda s,sid,v,arr: (s,sid,v,tf.squeeze(arr))).prefetch(tf.data.AUTOTUNE)

    def build_pairs(si,siid,vi,arri):
        return d2.map(lambda sj, sjid, vj, arrj: (si,sj,siid,sjid,vi,vj,arri,arrj))
    d3 = d2.interleave(build_pairs, cycle_length=8, block_length=1000, 
        num_parallel_calls=tf.data.AUTOTUNE,  deterministic=True)
    d3 = d3.filter(lambda si,sj,siid,sjid,vi,vj,arri,arrj: tf.less(siid,sjid))

    x = next(iter(d3))
    j = next(iter(d2))
    d4 = d3.map(lambda i,j,arri,arrj: (i,j,tf.einsum('ij,kj->ik', arri, arrj)))
    # round the similarity matrix to 3 decimal places
    d4 = d4.map(lambda i,j,sim: (i,j,tf.cast(tf.round(sim*1000),dtype=tf.int16)))

    def reshapeit(i,j,sim):
        ir = tf.repeat(i, tf.shape(j))
        jt = tf.tile(j, tf.shape(i))
        sim = tf.reshape(sim, [-1])
        indices = tf.where(sim > 600)
        ir = tf.squeeze(tf.gather(ir, indices))
        jt = tf.squeeze(tf.gather(jt, indices))
        sim = tf.squeeze(tf.gather(sim, indices))
        return ir,jt,sim

    d5 = d4.map(reshapeit)

    def eval_counts(i,j,sim):
        cindices = tf.where(i == j)
        ccounts = tf.unique_with_counts(tf.squeeze(tf.gather(sim,cindices)))
        return (ccounts, tf.unique_with_counts(sim))

    d6 = d5.map(eval_counts)
    # reduce d5 into counts of rows with each similarity
    
    batch = next(iter(d6))
    
    # make a dictionary taking the range from 600 to 1000 in steps of 1 to 0
    ccnt = {i:0 for i in np.arange(600,1001,1,dtype=np.int16)}
    tcnt = {i:0 for i in np.arange(600,1001,1,dtype=np.int16)}

    for batch in d6:
        csim, _, ccount = [x.numpy() for x in batch[0]]
        ccnt = {k: ccnt[k] + ccount[i] for i, k in enumerate(csim)}
        tsim, _, tcount = [x.numpy() for x in batch[1]]
        tcnt = {k: tcnt[k] + tcount[i] for i, k in enumerate(tsim)}
        
    # combine the ccnt and tcnt dictionaries into a dataframe
    emb, pid = (path.split('/')[2].split('=')[1], path.split('/')[3].split('=')[1])
    sim, C, T = (list(ccnt.keys()), list(ccnt.values()), [tcnt[x] for x in ccnt])
    df = pd.DataFrame({'pid': pid, 'embedding': emb, 'sim': sim, 'correct': C, 'N': T})

    df.to_csv(dfpath, mode='a', header=False, index=False)


pat = re.compile('.*/pid=[0-9.]+$')
tfpaths = (p.as_posix() for p in pl.Path('cache/chemprop_tfds').glob('**/') if pat.match(p.as_posix()))

dfpath = 'cache/evalsims.csv'

# delete dfpath if it already exists
if os.path.exists(dfpath):
    os.remove(dfpath)

# write a header line to dfpath
pd.DataFrame({'pid':[], 'embedding':[], 'sim':[], 'correct':[], 'N':[]}).to_csv(dfpath, index=False)

for path in tfpaths:
    build_property_evaluation_tfds(path)

# 1. rebuild the chemprop tfds into one large dataset for each embedding
# 2. reduce this dataset into
#    a. a dictionary of counts of correct/incorrect predictions for knn predictions
# How to test this?
# 1. build knn 
# 2. test knn on existing base paths
# 3. test knn on supervised model path