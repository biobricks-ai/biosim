import tensorflow as tf, os
import pathlib, sqlite3, math, numpy as np, timeit

# conn = sqlite3.connect('cache/cache.db')
# c = conn.cursor()
# c.execute("SELECT DISTINCT embedding FROM embeddings")
# embeddings = [x[0] for x in c.fetchall()]

# dims = {'bert':768, 'pubchem':881}

## Create tfrecords from the embeddings cache/cache.db sqlite table
for embedding in embeddings:
    
    pathlib.Path(f'cache/tfrecord/embedding-{embedding}').mkdir(parents=True, exist_ok=True)
    
    T = (tf.int32,tf.string,tf.string,tf.string)
    Q = f"""SELECT canonical_smiles, arrstr, embedding 
    FROM embeddings WHERE embedding = '{embedding}' ORDER BY RANDOM() LIMIT 10000"""
    
    dataset = tf.data.experimental.SqlDataset("sqlite", "cache/cache.db",Q,T)

    floats = dataset.map(lambda i,smi,arr,emb: 
        tf.reshape(tf.strings.to_number(tf.strings.split(arr, ",")), (-1, dims[embedding])))

    other = dataset.map(lambda i,smi,arr,emb: (i,smi,emb))

    dataset = tf.data.Dataset.zip((other,floats)).map(lambda x,y: (x[0],x[1],x[2],y))

    tf.data.Dataset.save(dataset, f'cache/tfrecord/embedding-{embedding}')


# THERE ARE EXACTLY 320146 UNIQUE SMILES IN THE DATASET
d1 = tf.data.Dataset.load('cache/tfrecord/embedding-bert')
d2 = d1.map(lambda i,s,e,arr: (i,arr)).batch(10000) # about 40 batches
d2 = d2.map(lambda i,arr: (i,tf.squeeze(arr)))
d2 = d2.prefetch(tf.data.AUTOTUNE)

d3 = d2.interleave(lambda i,arri: d2.map(lambda j,arrj: (i,j,arri,arrj)),
    cycle_length=8, block_length=1000, num_parallel_calls=tf.data.AUTOTUNE,
    deterministic=True)

d4 = d3.map(lambda i,j,arri,arrj: (i,j,tf.einsum('ij,kj->ik', arri, arrj)))

def reshapeit(i,j,sim):
    ir = tf.repeat(i, tf.shape(j))
    jt = tf.tile(j, tf.shape(i))
    sim = tf.reshape(sim, [-1])
    indices = tf.where(sim > 0.6)
    ir = tf.gather(ir, indices)
    jt = tf.gather(jt, indices)
    sim = tf.gather(sim, indices)
    return ir,jt,sim

d5 = d4.map(reshapeit)
# .unbatch().filter(lambda i,j,sim: sim > 0.9)

sqlite3.register_adapter(np.int32, lambda val: int(val))
sqlite3.register_adapter(np.int64, lambda val: int(val))
sqlite3.register_adapter(np.float32, lambda val: float(val))

def sqlstore(tuples):
    conn = sqlite3.connect('cache/test.db')
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS sims (i INTEGER, j INTEGER, sim REAL)")
    c.executemany("INSERT INTO sims VALUES (?,?,?)", tuples)
    conn.commit()
    conn.close()
    return True

batches = [x for x in d5][0]
batches = [x.numpy() for x in batches]
i,j,sim = batches
tuples = [(x[0],y[0],z[0]) for x,y,z in zip(i,j,sim)]
sqlstore(tuples)


def calcit():
    d5.save('cache/sims')
    print(os.system('du -h cache/sims'))
    os.system('rm -rf cache/sims')
    return True


# 1000 batch  = 10, 43, 93 seconds
# 10000 batch = 10, 43, 93 seconds
timeit.timeit(calcit, number=1)
 
for i,x in enumerate(d5):
    print(x[1])

d5 = d4.filter(lambda i,smi,smi2,emb,arr,arr2,sim: sim > 0.9).take(1000000)

d5.save('cache/sims')