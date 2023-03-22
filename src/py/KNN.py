import tensorflow as tf
import src.py.utilities as util

@tf.function
def knn_map(ai,aj,vj,k):
    nai = tf.linalg.l2_normalize(ai)
    naj = tf.linalg.l2_normalize(aj,axis=1)
    bsim = tf.tensordot(nai,tf.transpose(naj),axes=1)
    topk = tf.nn.top_k(bsim, k=k+1,sorted=True)
    maxidx,maxsim = topk.indices[1:], topk.values[1:]
    maxval = tf.gather(vj, indices=maxidx)
    return {"maxsim":maxsim, "maxval":maxval}

@tf.function
def vote(sims,vals):
    weights = tf.tensordot(sims,tf.cast(vals,dtype=tf.float32),axes=1)
    return tf.reduce_sum(weights)/tf.reduce_sum(sims)

def evaluate_knn(ds,k=5):
    d1N = util.count_and_collect(ds)
    cmp = ds.batch(d1N).cache() # THE BELOW CODE RELIES ON HAVING A SINGLE BATCH FOR CMP
    
    # create simval = {vi: value, maxsim: similarities of topk, maxval: values of topk}
    mksims = lambda ai,vi: cmp.map(lambda aj,vj: dict({"vi":vi},**knn_map(ai,aj,vj,k)))
    simval = ds.interleave(mksims, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    return simval.map(lambda x: {"prediction":vote(x['maxsim'],x['maxval']),"binvalue":x['vi']})