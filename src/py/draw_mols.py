from rdkit import Chem
from rdkit.Chem import Draw

def draw_mols(smiles,output):
  mols = [Chem.MolFromSmiles(smi) for smi in smiles]
  svg  = Draw.MolsToGridImage(mols,molsPerRow=2,useSVG=True)
  with open(output, "w") as text_file:
      text_file.write(svg)