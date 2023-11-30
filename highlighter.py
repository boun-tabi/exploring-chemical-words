import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path 
from argparse import ArgumentParser
from data import load_bdb, load_lit_pcba, load_protein_class
from word_identification import WordIdentifier

dataset_list = ['bdb', 'epigenetic-regulators', 'hydrolases', 'ion-channels',
                'membrane-receptors', 'other-enzymes', 'oxidoreductases', 'proteases', 
                'transcription-factors', 'transferases', 'transporters'] #, 'lit_pcba']

def identify_highlights(documents, vocabulary='chembl27_bpe_8K', return_highlights=True, threshold=10):
    chem_tokenizer = WordIdentifier.from_file(str(Path(__file__).parent / f'data/vocabs/{vocabulary}.json'))
    doc_to_tokens = {}
    vocab = set()
    for doc_id, smiles in documents.items():
        smiles_token_ids = sum(chem_tokenizer.identify_words(smiles, out_type='int'), [])
        doc_to_tokens[doc_id] = ' '.join([str(number) for number in smiles_token_ids])
        vocab |= set(smiles_token_ids)

    vocab = {str(number) for number in vocab}

    corpus = list(doc_to_tokens.values())
    # TODO: max_df significantly affects highlighted words
    vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b', max_df=len(documents)//5*3, sublinear_tf=True)
    vectors = vectorizer.fit_transform(corpus).toarray()

    columns = [chem_tokenizer.tokenizer.id_to_token(int(token_id)) for token_id in vectorizer.get_feature_names()]
    tf_idf = pd.DataFrame(vectors, columns=columns, index=doc_to_tokens.keys())
    if return_highlights:	
        # TODO: Check top n%
        highlighted_words = tf_idf.apply(lambda x: pd.Series(x.nlargest(min(threshold, len(x[x!=0])) if type(threshold) == int else int(len(x[x!=0])*threshold)).index.values), axis=1)
        return tf_idf, highlighted_words.fillna('')
    else:
        return tf_idf

    

if __name__ == '__main__': 
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument('--vocabulary', type=str)
    parser.add_argument("--binders", type=str, choices=['active', 'inactive'])
    parser.add_argument("--threshold", type=float)
    args = parser.parse_args()

    if args.dataset is None: 
        datasets = dataset_list
    else: 
        datasets = [args.dataset]
    if args.vocabulary is None:
        vocab_list = [p.stem for p in Path('data/vocabs/').iterdir()]
    else:
        vocab_list = [args.vocabulary]
    if args.binders is None: 
        binders = ['active', 'inactive']
    else:
        binders = [args.binders]
        
    for dataset in datasets: 
        for vocabulary in vocab_list: 
            for binder in binders:
                if dataset == 'lit_pcba':
                    documents, doc_to_name, doc_to_count = load_lit_pcba(binders=binder, return_mappings=True)
                elif dataset == 'bdb':
                    documents, doc_to_name, doc_to_count = load_bdb(binders=binder, return_mappings=True)
                else: 
                    documents, doc_to_name, doc_to_count = load_protein_class(dataset, binders=binder, return_mappings=True)

                tf_idf, highlighted_words = identify_highlights(documents, vocabulary, return_highlights=True)

                highlighted_words.insert(0, 'doc_name', highlighted_words.index.map(doc_to_name))
                highlighted_words.insert(1, 'doc_count', highlighted_words.index.map(doc_to_count))
                highlighted_words.index.name = 'doc_id'
                highlighted_words.fillna('', inplace=True)
                for ix1 in range(len(highlighted_words)):
                    for ix2 in range(len(highlighted_words.columns)):
                        highlighted_words.iloc[ix1, ix2] = ' ' + str(highlighted_words.iloc[ix1, ix2])
                        
                output_dir = Path('results/datasets') / dataset / vocabulary / binder
                output_dir.mkdir(exist_ok=True, parents=True)
                tf_idf.to_csv(output_dir / 'tf_idf.csv')
                highlighted_words.to_excel(output_dir /'highlighted_words.xlsx')