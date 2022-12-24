# use the base embeddings and triplet methods 
# to build property specific embeddings 

import tensorflow as tf, os
import pathlib, sqlite3, math, numpy as np, timeit


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