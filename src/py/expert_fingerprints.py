import rdkit
from rdkit import Chem

def pubchem2d(smiles):
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    fps  = [rdkit.Chem.RDKFingerprint(mol) for mol in mols]
    return fps