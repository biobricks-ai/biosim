from time import time
import tensorflow as tf, transformers, pandas as pd, numpy as np
import biobricks as bb, pyarrow as pa, os
import contextlib

# BIOBRICKS ================================================================
os.environ['BBLIB'] = "/home/tomlue/BBLIB"
bb.load('chemharmony')
substances = "/home/tomlue/BBLIB/biobricks-ai/chemharmony/brick/harmonized/substances.parquet"
substances = pa.parquet.read_table(substances).to_pandas()
smiles = [x for x in substances['smiles'].unique().tolist() if x is not None]

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
    
## CHEMBERT ================================================================
hface = "submodules/ChemBERTa-zinc-base-v1"
tok = transformers.AutoTokenizer.from_pretrained(hface)
brt = transformers.TFAutoModel.from_pretrained(hface,from_pt=True)

def embed_chembert(smi):
    inp = tok.batch_encode_plus(smi, padding="max_length")
    return brt(tf.constant(inp['input_ids']), tf.constant(inp['attention_mask']))[1]

# create batches of 1k values from the smiles list
bds = tf.data.Dataset.from_tensor_slices({"smiles":smiles}).batch(100)
def embed_generator():
    nb,t0 = bds.cardinality().numpy(), time()
    for i,x in enumerate(bds):
        print(i)
        smi = [x.decode() for x in x['smiles'].numpy()]
        x['emb_pubchem2d'] = embed_pubchem2d(smi)
        x['emb_chembert'] = embed_chembert(smi)
        print(f"{i} / {nb} in {time()-t0:.2f}s rem: {(time()-t0)/(i+1)*(nb-i-1):.2f}s")
        yield x

types = {'smiles': tf.TensorSpec(shape=(100), dtype=tf.string),}
types['emb_chembert'] = tf.TensorSpec(shape=(100,768), dtype=tf.float32)
types['emb_pubchem2d'] = tf.TensorSpec(shape=(100,881), dtype=tf.float32)
embed_ds = tf.data.Dataset.from_generator(embed_generator, output_signature=types)
embed_ds.save("cache/tmp/emb_smiles/ds2")
