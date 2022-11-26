import torch
import transformers

import numpy
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, BertModel, AutoModel, RobertaModel

tok = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
mod = RobertaModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

comp_a = "CCCCC[C@@H](Br)CC"
comp_b = "C[C@H]1/C=C/C=C(\C(=O)NC2=C(C(=C3C(=C2O)C(=C(C4=C3C(=O)[C@](O4)(O/C=C/[C@@H]([C@H]([C@H]([C@@H]([C@@H]([C@@H]([C@H]1O)C)O)C)OC(=O)C)C)OC)C)C)O)O)/C=N/N5CCN(CC5)C)/C"

comps = [comp_a,comp_b]
inputs = tok(comps, padding=True, truncation=True, return_tensors="pt")

outputs = mod(**inputs)

noutput = torch.max(torch.norm(outputs[0],dim=-1))
out2 = torch.divide(outputs[0], noutput)

a = outputs[1][0]
b = outputs[1][1]
torch.cosine_similarity(a,b,dim=0)

# TODO fine tune roberta for biosim