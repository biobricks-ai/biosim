import deepchem as dc
import numpy as np
import pandas as pd
import tensorflow as tf

from rdkit import Chem

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
smod = dc.models.seqtoseq.AspuruGuzikAutoEncoder(tokens, max_length)

def generate_sequences(smiles, epochs):
  for i in range(epochs):
    print(i)
    smod.encoder.save("model/encoder.hdf5")
    smod.decoder.save("model/decoder.hdf5")
    for s in smiles:
      yield (s, s)

smod.fit_sequences(generate_sequences(train_smiles, 100))


def norm_emb(smi):
  emb = s.predict_embeddings(smi)
  norm = tf.reshape(tf.norm(emb,axis=1),(len(smi),1))
  return tf.divide(emb, norm)

ancsmi = train_smiles[0:1000]
anc  = embed(ancsmi,bmod)

smi = train_smiles[1000:100000]
emb = embed(smi,bmod)

mul  = tf.matmul(anc,tf.transpose(emb))

mols = []
for i in range(5):
  maxj = tf.argmax(mul[i,])
  minj = tf.argmin(mul[i,])
  mol1 = Chem.MolFromSmiles(ancsmi[i])
  mol2 = Chem.MolFromSmiles(smi[maxj])
  mol3 = Chem.MolFromSmiles(smi[minj])

  mols.append(mol1)
  mols.append(mol2)
  # mols.append(mol3)

from rdkit.Chem import Draw
svg = Draw.MolsToGridImage(mols,molsPerRow=2,useSVG=True)
with open("Output.svg", "w") as text_file:
    text_file.write(svg)

pred1 = s.predict_from_sequences(train_smiles, beam_width=1)
pred4 = s.predict_from_sequences(train_smiles, beam_width=4)

embeddings = s.predict_embeddings(train_smiles[0:5])

# assay 
1. 
# pred1e = s.predict_from_embeddings(embeddings, beam_width=1)
# psmi = [''.join(p) for p in pred1e]

# pred4e = s.predict_from_embeddings(embeddings, beam_width=4)
# psmi = [''.join(p) for p in pred4e]


# for i in range(len(train_smiles)):
#   assert pred1[i] == pred1e[i]
#   assert pred4[i] == pred4e[i]

# def test_variational(self):
#   """Test using a SeqToSeq model as a variational autoenconder."""

#   sequence_length = 10
#   tokens = list(range(10))
#   s = dc.models.SeqToSeq(
#       tokens,
#       tokens,
#       sequence_length,
#       encoder_layers=2,
#       decoder_layers=2,
#       embedding_dimension=128,
#       learning_rate=0.01,
#       variational=True)

# # Actually training a VAE takes far too long for a unit test.  Just run a
# # few steps of training to make sure nothing crashes, then check that the
# # results are at least internally consistent.

#   s.fit_sequences(generate_sequences(sequence_length, 1000))
#   for sequence, target in generate_sequences(sequence_length, 10):
#     pred1 = s.predict_from_sequences([sequence], beam_width=1)
#     embedding = s.predict_embeddings([sequence])
#     assert pred1 == s.predict_from_embeddings(embedding, beam_width=1)