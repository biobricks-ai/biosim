import tensorflow as tf, os, pathlib
import pathlib as pl, math, numpy as np, timeit
import re, tensorflow as tf, pandas as pd

def build_property_evaluation_tfds(path):

    d1 = tf.data.Dataset.load(path)
    d2 = d1.map(lambda a,p,e,s,arr,v: (s,tf.strings.to_hash_bucket_fast(s,100e6),v,arr)).batch(10000)
    d2 = d2.map(lambda s,v,arr: (s,v,tf.squeeze(arr))).prefetch(tf.data.AUTOTUNE)

    def build_pairs(si,vi,arri):
        return d2.map(lambda sj, vj, arrj: (si,sj,vi,vj,arri,arrj))
    d3 = d2.interleave(build_pairs, cycle_length=8, block_length=1000, 
        num_parallel_calls=tf.data.AUTOTUNE,  deterministic=True)
    d3 = d3.filter(lambda si,sj,vi,vj,arri,arrj: tf.equal(si < sj))

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
tfpaths = (p.as_posix() for p in pl.Path('cache/tfrecord').glob('**/') if pat.match(p.as_posix()))

dfpath = 'cache/evalsims.csv'

# delete dfpath if it already exists
if os.path.exists(dfpath):
    os.remove(dfpath)

# write a header line to dfpath
pd.DataFrame({'pid':[], 'embedding':[], 'sim':[], 'correct':[], 'N':[]}).to_csv(dfpath, index=False)

for path in tfpaths:
    build_property_evaluation_tfds(path)