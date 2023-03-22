from time import time
import tensorflow as tf, transformers, pandas as pd, numpy as np
import biobricks as bb, pyarrow as pa, os
import contextlib

# BIOBRICKS ================================================================

activities = pa.parquet.read_table("cache/tmp/activities.parquet").to_pandas()
smiles = [x for x in activities['smiles'].unique().tolist()]
smiles = [x for x in smiles if x is not None and len(x) < 200]

# PUBCHEM2D ================================================================
import jnius_config 
jnius_config.set_classpath('/home/tomlue/git/ai.biobricks/biosim/src/scala/fingerprinters.jar')

from jnius import autoclass

sb = autoclass("org.openscience.cdk.silent.SilentChemObjectBuilder")
parser = autoclass("org.openscience.cdk.smiles.SmilesParser")(sb.getInstance())
finger = autoclass("org.openscience.cdk.fingerprint.PubchemFingerprinter")(sb.getInstance())

def fp2array(fp,arr=[],i=-1):
    i = fp.nextSetBit(i+1)
    return fp2array(fp, arr + [i], i) if i > -1 else arr
    
def smi2fp(smi):
    res = []
    with contextlib.suppress(Exception):
        mol = parser.parseSmiles(smi)
        res = fp2array(finger.getFingerprint(mol))
    return tf.reduce_sum(tf.one_hot(res,881),axis=0)

def embed_pubchem2d(smi):
    return tf.concat([[smi2fp(x) for x in smi]],axis=1)

# CHEMBERT ================================================================
# TODO come back and try with chembert
# hface = "submodules/ChemBERTa-zinc-base-v1"
# tok = transformers.AutoTokenizer.from_pretrained(hface)
# brt = transformers.TFAutoModel.from_pretrained(hface,from_pt=True)

# def embed_chembert(smi):
#     inp = tok.batch_encode_plus(smi, padding="longest")
#     ids = tf.constant(inp['input_ids'])
#     att = tf.constant(inp['attention_mask'])
#     return brt(ids,att)[1]

# # create batches of 1k values from the smiles list
bds = tf.data.Dataset.from_tensor_slices({"smiles":smiles}).batch(100)

def embed_generator():
    nb,t0 = bds.cardinality().numpy(), time()
    for i,x in enumerate(bds):
        smi = [x.decode() for x in x['smiles'].numpy()]
        x['emb_pubchem2d'] = embed_pubchem2d(smi)
        # x['emb_chembert'] = embed_chembert(smi)
        yield x

types = {'smiles': tf.TensorSpec(shape=(None), dtype=tf.string),}
# types['emb_chembert'] = tf.TensorSpec(shape=(None,768), dtype=tf.float32)
types['emb_pubchem2d'] = tf.TensorSpec(shape=(None,881), dtype=tf.float32)
embed_ds = tf.data.Dataset.from_generator(embed_generator, output_signature=types)

# ACTIVITIES ================================================================
def activities_generator():
    nb,t0 = bds.cardinality().numpy(), time()
    for i,batch in enumerate(embed_ds):
        smiles = [x.decode() for x in batch['smiles'].numpy()]
        # emb_chembert = batch['emb_chembert'].numpy()
        emb_pubchem2d = batch['emb_pubchem2d'].numpy()
        df = pd.DataFrame({'smiles':smiles,
            # 'emb_chembert':emb_chembert.tolist(),
            'emb_pubchem2d':emb_pubchem2d.tolist()})
        out = activities.merge(df,how='inner',on='smiles').to_dict('list')
        print(f"{i} / {nb} in {time()-t0:.2f}s rem: {(time()-t0)/(i+1)*(nb-i-1):.2f}s")
        res = {k:tf.constant(v) for k,v in out.items()}
        # select smiles, emb_chembert, emb_pubchem2d from res
        yield {k:res[k] for k in ['smiles','pid','pidnum','value','normvalue','binvalue',
                                #   'emb_chembert',
                                  'emb_pubchem2d']}

types = {
    'smiles': tf.TensorSpec(shape=(None), dtype=tf.string),
    'pid': tf.TensorSpec(shape=(None), dtype=tf.string),
    'pidnum': tf.TensorSpec(shape=(None), dtype=tf.float32),
    'value': tf.TensorSpec(shape=(None), dtype=tf.float32),
    'normvalue': tf.TensorSpec(shape=(None), dtype=tf.float32),
    'binvalue': tf.TensorSpec(shape=(None), dtype=tf.float32),
    # 'emb_chembert': tf.TensorSpec(shape=(None,768), dtype=tf.float32),
    'emb_pubchem2d': tf.TensorSpec(shape=(None,881), dtype=tf.float32)
}

dsactivities = tf.data.Dataset.from_generator(activities_generator, output_signature=types)
dsactivities.unbatch().save("cache/tfdatasets/activity_embedding")