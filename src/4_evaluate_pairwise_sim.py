import tensorflow as tf
import pandas as pd
import itertools, math
from alive_progress import alive_bar
import tqdm
import src.py.utilities as util

# SIMILARITY EVALUATION ========================================================
activities = tf.data.Dataset.load("cache/tfdatasets/activity_embedding")
activities = activities.map(lambda d: dict(d,**{'pidnum':tf.cast(d['pidnum'],tf.int32)}))

chunk = math.ceil(activities.cardinality().numpy()/10)
holdout = activities.skip(9*chunk).take(chunk)

pidnums = holdout.map(lambda x: x['pidnum']).unique().cache()
pidnums = [x for x in pidnums.as_numpy_iterator()]
maxpidnum = max(pidnums)
embeddings = ["emb_pubchem2d","emb_chembert","emb_sup"]

# Create pandas DF for results
sims = [x for x in range(1001)]
prod = itertools.product(pidnums,embeddings,sims)
df = pd.DataFrame(prod,columns=["pidnum","emb","sim"])
df["ntrue"], df["total"] = (0,0)
df.set_index(["pidnum","emb","sim",],inplace=True)
df.sort_index(inplace=True)

for pidnum, emb in itertools.product(pidnums,embeddings):
    print(f"starting {pidnum} / {emb}")
    act = activities.filter(lambda x: x['pidnum'] == pidnum)
    act = act.batch(20000).map(lambda x: dict(x,**{emb:tf.linalg.l2_normalize(x[emb],axis=1)})).cache()
    hld = holdout.filter(lambda x: x['pidnum'] == pidnum)
    hld = hld.map(lambda x: dict(x,**{emb:tf.linalg.l2_normalize(x[emb])})).cache()
    
    Nhld = util.count_and_collect(hld)
    Nact = util.count_and_collect(act)
    
    def build_pairs(x):
        sim = lambda y: tf.cast(1000*tf.matmul(y[emb],tf.expand_dims(x[emb],1)),tf.int32)
        com = lambda y: tf.equal(x['binvalue'],y['binvalue'])
        return act.map(lambda y: (com(y), tf.squeeze(sim(y))))

    sim = hld.interleave(build_pairs, num_parallel_calls=tf.data.AUTOTUNE)
    
    with alive_bar(Nhld*Nact) as bar:
        for i,x in enumerate(sim.as_numpy_iterator()):
            match,sim = x
            istrue = tf.squeeze(tf.gather(sim,tf.where(match)))
            ntrues = tf.unique_with_counts(istrue)
            ntotal = tf.unique_with_counts(sim) 
            df.loc[(pidnum,emb,ntrues.y.numpy()),"ntrue"] += ntrues.count.numpy()
            df.loc[(pidnum,emb,ntotal.y.numpy()),"total"] += ntotal.count.numpy()
            bar()

df.to_csv(f"cache/eval/similarity.csv")