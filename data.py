import pandas as pd
from pathlib import Path 

def load_bdb(binders='active', return_mappings=False):
	dataset = Path(__file__).parent / 'data/bdb'
	pfam = pd.read_csv(dataset / 'pfams.csv')
	bdb = pd.read_csv(dataset / 'interactions.csv')
	threshold = '>' if binders == 'active' else '<'
	bdb_filtered = bdb.query(f'affinity_score {threshold} 7')
	prot_to_family = pfam.groupby('prot_id')['pfam_id'].first().to_dict()
	bdb_filtered['pfam_id'] = bdb_filtered['prot_id'].map(prot_to_family)
	bdb_filtered['pfam_id'] = bdb_filtered['pfam_id'].str.split(';')
	bdb_filtered = bdb_filtered.explode('pfam_id')
	pfam_to_smiles = bdb_filtered.groupby('pfam_id')['smiles'].apply(list).to_dict()
	if return_mappings:
		# Dataframe is updated with bdb_filtered to account for only filtered interactions.
		family_to_prot = bdb_filtered.groupby('pfam_id')['prot_id'].apply(list).to_dict()
		family_to_count = {family: str(len(set(prot))) for family, prot in family_to_prot.items()}
		print('Number of proteins without family', pfam.isna().sum())
		pfam.dropna(inplace=True)
		pfam_ids = ';'.join(pfam['pfam_id'].tolist())
		pfam_names = ';'.join(pfam['pfam_name'].tolist())
		family_to_name = {pid: pname for pid, pname in zip(pfam_ids.split(';'), pfam_names.split(';'))}
		return pfam_to_smiles, family_to_name, family_to_count
	else:
		return pfam_to_smiles

def load_lit_pcba(binders='active', return_mappings=False):
	lit_pcba_path = Path(__file__).parent / 'data/lit-pcba'
	files = [f for f in lit_pcba_path.glob(f'*/*_{binders}_*.smi')]
	interactions = pd.concat([pd.read_csv(f, sep=' ', names=['smiles', 'ligand_id'], header=None).assign(prot_id=f.parent.name) for f in files])
	prot_to_smiles = interactions.groupby('prot_id')['smiles'].apply(list).to_dict()
	prot_to_count = {prot: str(len(smiles)) for prot, smiles in prot_to_smiles.items()}
	if return_mappings:
		return prot_to_smiles, {prot: prot for prot in prot_to_smiles}, prot_to_count
	else:
		return prot_to_smiles


def load_protein_class(family, binders='active', return_mappings=False):
	folder_path = Path(__file__).parent / 'data/protein-class'
	compound_sequences = pd.read_csv(folder_path / 'compound_sequences.csv')
	train = pd.read_csv(folder_path / f'{family}_train.tsv', sep='\t')
	test = pd.read_csv(folder_path / f'{family}_test.tsv', sep='\t')
	threshold_value = train['pchembl_value'].median()
	threshold_sign = '>' if binders == 'active' else '<'
	interactions = pd.concat([train, test])
	filtered_interactions = interactions.query(f'pchembl_value {threshold_sign} {threshold_value}')
	chem_to_smiles = compound_sequences.groupby('compound_id')['canonical_smiles'].first().to_dict()
	filtered_interactions['smiles'] = filtered_interactions['compound_id'].map(chem_to_smiles)
	filtered_interactions.dropna(inplace=True)
	prot_to_smiles = filtered_interactions.groupby('target_id')['smiles'].apply(list).to_dict()
	prot_to_count = {prot: str(len(smiles)) for prot, smiles in prot_to_smiles.items()}
	if return_mappings: 
		return prot_to_smiles, {prot: prot for prot in prot_to_smiles}, prot_to_count
	else:
		return prot_to_smiles
