import deepchem as dc
import numpy as np
import pandas as pd

"""
BUILDING A VAE 
https://github.com/deepchem/deepchem/blob/9664aeab16940e1ee46030cac08d90e12aa24751/deepchem/models/seqtoseq.py
"""

df = pd.read_csv("cache/train.csv")
df2 = df[df["canonical_smiles"].apply(lambda x: len(x) < 200)]
train_smiles = df2['canonical_smiles'].unique().tolist()[1:100000]

tokens = set()
for s in train_smiles:
  tokens = tokens.union(set(c for c in s))

tokens = sorted(list(tokens))
max_length = max(len(s) for s in train_smiles) + 1
s = dc.models.seqtoseq.AspuruGuzikAutoEncoder(tokens, max_length)

def generate_sequences(smiles, epochs):
  for i in range(epochs):
    print(i)
    for s in smiles:
      yield (s, s)

s.fit_sequences(generate_sequences(train_smiles, 100))

# Test it out.
pred1 = s.predict_from_sequences(train_smiles, beam_width=1)
pred4 = s.predict_from_sequences(train_smiles, beam_width=4)

embeddings = s.predict_embeddings(train_smiles[0:5])

pred1e = s.predict_from_embeddings(embeddings, beam_width=1)
pred4e = s.predict_from_embeddings(embeddings, beam_width=4)

for i in range(len(train_smiles)):
  assert pred1[i] == pred1e[i]
  assert pred4[i] == pred4e[i]

def test_variational(self):
  """Test using a SeqToSeq model as a variational autoenconder."""

  sequence_length = 10
  tokens = list(range(10))
  s = dc.models.SeqToSeq(
      tokens,
      tokens,
      sequence_length,
      encoder_layers=2,
      decoder_layers=2,
      embedding_dimension=128,
      learning_rate=0.01,
      variational=True)

# Actually training a VAE takes far too long for a unit test.  Just run a
# few steps of training to make sure nothing crashes, then check that the
# results are at least internally consistent.

  s.fit_sequences(generate_sequences(sequence_length, 1000))
  for sequence, target in generate_sequences(sequence_length, 10):
    pred1 = s.predict_from_sequences([sequence], beam_width=1)
    embedding = s.predict_embeddings([sequence])
    assert pred1 == s.predict_from_embeddings(embedding, beam_width=1)