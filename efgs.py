import swifter
import json
import pandas as pd

from tqdm import tqdm, tqdm_notebook
from tqdm import tqdm
tqdm.pandas()


def get_frags(s): 
	from EFGs import mol2frag
	from rdkit import Chem
	try:
		return mol2frag(Chem.MolFromSmiles(s))
	except:
		return None, None 

chembl = pd.read_csv('data/chembl27.csv')
chembl['efgs'] = chembl['canonical_smiles'].swifter.apply(lambda s: get_frags(s))
chembl.to_csv('data/chembl27_efg.csv', index=False)
chembl['functional_groups'] = chembl['efgs'].apply(lambda p: p[0])
vocab  = {'type': 'efg', 'data': 'chembl27', 'vocab': chembl['functional_groups'].drop_duplicates().tolist()}
with open('data/chembl27_efg.json', 'w')  as f:
	f.write(json.dumps(vocab))