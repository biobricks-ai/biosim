import deepchem as dc
import numpy as np
import pandas as pd
import tensorflow as tf

from rdkit import Chem

import importlib.util
import sys
spec = importlib.util.spec_from_file_location("seq2seq", "src/py/seq2seq.py")
seq2seq = importlib.util.module_from_spec(spec)
sys.modules["seq2seq"] = seq2seq
spec.loader.exec_module(seq2seq)
seq2seq.AspuruGuzikAutoEncoder
"""
BUILDING A VAE 
https://github.com/deepchem/deepchem/blob/9664aeab16940e1ee46030cac08d90e12aa24751/deepchem/models/seqtoseq.py
"""

df = pd.read_csv("cache/train.csv")
df2 = df[df["canonical_smiles"].apply(lambda x: len(x) < 200)]
train_smiles = df2['canonical_smiles'].unique().tolist()
len(train_smiles)

tokens = set()
for s in train_smiles:
  tokens = tokens.union(set(c for c in s))

tokens = sorted(list(tokens))
max_length = max(len(s) for s in train_smiles) + 1
smod = seq2seq.AspuruGuzikAutoEncoder(tokens, max_length,dropout=0.1)
smod.model.compile(metrics="loss")

def generate_sequences(smiles, epochs):
  for i in range(epochs):
    # smod.encoder.save("model/encoder.hdf5")
    # smod.decoder.save("model/decoder.hdf5")
    print(i)
    for s in smiles:
      yield (s, s)

res = smod.fit_sequences(generate_sequences(train_smiles[0:1000], 10))

emb = smod.predict_embeddings(train_smiles[0:99])

import keras


def predict_embeddings(smod, sequences):
  """Given a set of input sequences, compute the embedding vectors."""
  result = []
  for batch in bmod.batch_sequences(sequences):
    features = bmod.create_input_array(batch)
    indices = np.array([(i, len(batch[i]) if i < len(batch) else 0)
                        for i in range(bmod.batch_size)])
    embeddings = bmod.predict_on_generator(
        [[(features, indices, np.array(smod.get_global_step())), None, None]],
        outputs=smod._embedding)
    for i in range(len(batch)):
      result.append(embeddings[i])
  return np.array(result, dtype=np.float32)


  




from deepchem.models.seqtoseq import SeqToSeq

def create_input_array(sequences,tok_dict):
  """Create the array describing the input sequences for a batch."""
  lengths = [len(x) for x in sequences]
  sequences = [reversed(s) for s in sequences]
  
  ntoks = len(smod._input_tokens)
  padsize = 200
  features = np.zeros((len(sequences),padsize, ntoks),dtype=np.float32)
  
  for i, sequence in enumerate(sequences):
    for j, token in enumerate(sequence):
      features[i, j, tok_dict[token]] = 1
  
  features[np.arange(len(sequences)), lengths, ntoks-1] = 1
  return features

def embed(smi,smod):
  features = create_input_array(smi,smod._input_dict)

  inp = bmod.encoder.inputs[0]
  out = bmod.encoder.layers[-3]
  mod = tf.keras.Model(inputs=[inp],outputs=out.output)

  emb  = mod.predict(features)
  norm = tf.reshape(tf.norm(emb,axis=1),(len(emb),1))
  return tf.divide(emb,norm)




emb  = embed(['c1ccccc1','c1ccccc1',smi[1]],smod)
tf.matmul(emb,tf.transpose(emb))

emb = embed([train_smiles[0],train_smiles[0],train_smiles[1]],bmod)
tf.matmul(emb,tf.transpose(emb))

emb = bmod.predict_embeddings([train_smiles[0],train_smiles[0],train_smiles[1]])
norm = tf.reshape(tf.norm(emb,axis=1),(len(emb),1))
emb = tf.divide(emb,norm)

tf.losses.cosine_similarity(emb[0,],emb[1,])

