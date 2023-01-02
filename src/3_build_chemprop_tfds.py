import tensorflow as tf, os
import pathlib, sqlite3, math, numpy as np, timeit

# CREATE EMBEDDING, PROPERTY, VALUE TABLE =================================

conn = sqlite3.connect('cache/cache.db')
c = conn.cursor()

c.execute("SELECT DISTINCT embedding FROM embeddings")
embeddings = [x[0] for x in c.fetchall()]

c.execute("SELECT DISTINCT property_id FROM activities")
properties = [x[0] for x in c.fetchall()]

dims = {'bert':768, 'pubchem':881}

for emb in embeddings:
  for prop in properties:

    path = pathlib.Path(f'cache/chemprop_tfds/embedding={emb}/pid={prop}')
    os.system(f'rm -r {path}') if path.exists() else None    
    path.mkdir(parents=True, exist_ok=True)
    
    # aid, pid, embedding, smiles, array, value
    T = (tf.int32,tf.string,tf.string,tf.string,tf.string,tf.int32)
    Q = f"""SELECT a.activity_id, a.property_id, e.embedding, a.canonical_smiles, e.arrstr, a.value
        FROM embeddings e inner join activities a ON e.canonical_smiles = a.canonical_smiles
        WHERE e.embedding = '{emb}' and a.property_id = '{prop}'"""
    
    dataset = tf.data.experimental.SqlDataset("sqlite", "cache/cache.db",Q,T)

    dim = dims[emb]
    def str_to_floats(arrstr):
        return tf.reshape(tf.strings.to_number(tf.strings.split(arrstr, ",")), (-1, dim))

    floats = dataset.map(lambda i,pid,emb,smi,arr,val: (i,pid,emb,smi,str_to_floats(arr),val))
    tf.data.Dataset.save(floats, str(path))

