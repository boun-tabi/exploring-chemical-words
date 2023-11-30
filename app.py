import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
import mols2grid
from rdkit import Chem
from pathlib import Path
from chembl_webresource_client.new_client import new_client
from data import load_bdb, load_lit_pcba
from highlighter import identify_highlights

@st.cache(allow_output_mutation=True)
def get_data(dataset_name):
    return load_bdb(return_mappings=True) if dataset_name == 'BDB' else load_lit_pcba(return_mappings=True)
   
st.title('Chemical Words in Spotlight')

dataset_name = st.radio("Dataset", ('BDB', 'Lit-PCBA')) #, 'Protein-Class'))
# if dataset_name == 'Protein-Class':
#    dataset_name = st.radio("Class", ('epigenetic-regulators', 'hydrolases', 'ion-channels', 'membrane-receptors', 
#       'other-enzymes', 'oxidoreductases', 'proteases', 'transcription-factors', 'transferases', 'transporters'))
documents, doc_to_name, doc_to_count = get_data(dataset_name)
tf_idf, highlighted_words = identify_highlights(documents, return_highlights=True)
highlighted_words.fillna('', inplace=True)
st.write('## Highlights for PFAM/Protein')

chosen_doc = st.selectbox( 'PFAM/Protein', list([f'{d}: {doc_to_name[d]}' for d in documents]))
chosen_doc_id = chosen_doc.split(':')[0]
chosen_highlights = highlighted_words.loc[chosen_doc_id]
chosen_highlights = chosen_highlights[chosen_highlights != ''].to_frame()
chosen_highlights['valid'] = chosen_highlights[chosen_doc_id].apply(lambda w:  True if Chem.MolFromSmiles(w) is not None else False)
st.write(chosen_highlights)

st.write('### Strong binders')
df_binders = pd.DataFrame.from_dict(documents[chosen_doc_id])
df_binders.columns = ['SMILES']
st.dataframe(df_binders, use_container_width=True)

raw_html = mols2grid.display(df_binders,
   subset=['img', 'SMILES'],
   tooltip=["SMILES"],
   tooltip_trigger="click hover",
   )._repr_html_()
components.html(raw_html, width=900, height=600, scrolling=True)

st.write('### Drugs and Clinical Trials')

targets_api = new_client.target
mechanism_api = new_client.mechanism
molecule_api = new_client.molecule

if dataset_name == 'Lit-PCBA':
   chembl_target = targets_api.filter(target_synonym__iexact=chosen_doc_id.split('_')[0]).only(['target_chembl_id', 'organism', 'pref_name', 'target_type'])[0]
   st.write(chembl_target['target_chembl_id'])
   approved_drugs = mechanism_api.filter(target_chembl_id=chembl_target['target_chembl_id'])
   df_approved_drugs = pd.DataFrame(approved_drugs) 
   if not df_approved_drugs.empty:
      mols = molecule_api.filter(molecule_chembl_id__in=[drug['molecule_chembl_id'] for drug in approved_drugs]).only(['molecule_chembl_id', 'molecule_structures'])
      mol_records = [(m['molecule_chembl_id'], m['molecule_structures']['canonical_smiles']) for m in mols if m['molecule_structures'] is not None]
      df_mol_records = pd.DataFrame.from_records(mol_records, columns=['ChEMBL ID', 'SMILES'])
      df_output = pd.merge(df_mol_records, df_approved_drugs, left_on='ChEMBL ID', right_on='molecule_chembl_id')
      tooltip = ["ChEMBL ID", "SMILES", 'action_type', 'max_phase', 'mechanism_of_action']
   else:
      st.write('No data found')
elif dataset_name == 'BDB': 
   dataset = Path(__file__).parent / 'data/bdb'
   pfam = pd.read_csv(dataset / 'pfams.csv')
   proteins = pfam[pfam['pfam_id'] == chosen_doc_id]['prot_id'].tolist()
   all_drugs = []
   for prot in proteins: 
      chembl_target = targets_api.get(target_components__accession=prot).only(
       "target_chembl_id", "organism", "pref_name", "target_type")[0]
      approved_drugs = mechanism_api.filter(target_chembl_id=chembl_target['target_chembl_id'])
      df_approved_drugs = pd.DataFrame(approved_drugs)
      if not df_approved_drugs.empty:
         mols = molecule_api.filter(molecule_chembl_id__in=[drug['molecule_chembl_id'] for drug in approved_drugs]).only(['molecule_chembl_id', 'molecule_structures'])
         mol_records = [(m['molecule_chembl_id'], m['molecule_structures']['canonical_smiles']) for m in mols if m['molecule_structures'] is not None]
         df_mol_records = pd.DataFrame.from_records(mol_records, columns=['ChEMBL ID', 'SMILES'])
         df_merged = pd.merge(df_mol_records, df_approved_drugs, left_on='ChEMBL ID', right_on='molecule_chembl_id')
         all_drugs.append(df_merged.assign(uniprot=prot, target=chembl_target['target_chembl_id'], target_name=chembl_target['pref_name']))
   df_output = pd.concat(all_drugs)
   tooltip = ["ChEMBL ID", "SMILES", 'action_type', 'max_phase', 'mechanism_of_action', 'uniprot', 'target_name']

raw_drug_html = mols2grid.display(df_output,
   subset=['img', 'ChEMBL ID', 'action_type'],
   tooltip=tooltip,
   tooltip_trigger="click hover",
   )._repr_html_()
components.html(raw_drug_html, width=900, height=600, scrolling=True)