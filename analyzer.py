import pandas as pd
import re
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pylcs

from scipy.stats import ranksums, zscore
from tokenizers import Tokenizer
from rdkit import Chem
from tqdm import tqdm
from argparse import ArgumentParser
from pathlib import Path 

from highlighter import identify_highlights
from data import load_bdb, load_lit_pcba, load_protein_class

tqdm.pandas()
sns.set(font_scale=1.4)

data_loader = {'bdb': load_bdb, 'lit_pcba': load_lit_pcba}
dataset_list = ['bdb', 'epigenetic-regulators', 'hydrolases', 'ion-channels',
                'membrane-receptors', 'other-enzymes', 'oxidoreductases', 'proteases', 
                'transcription-factors', 'transferases', 'transporters', 'lit_pcba']

SMI_REGEX_PATTERN = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#||\+|\\\\\/|:||@|\?|>|\*|\$|\%[0–9]{2}|[0–9])"
regex = re.compile(SMI_REGEX_PATTERN)

def conduct_significance_analysis(dataset, vocabulary, export=True):

    # if dataset == 'bdb': 
    #     actives = load_bdb(binders='active')
    #     inactives = load_bdb(binders='inactive')
    # elif dataset == 'lit_pcba': 
    #     actives = load_lit_pcba(binders='active')
    #     inactives = load_lit_pcba(binders='inactive')
    # else: 
    #     actives = load_protein_class(dataset, binders='active')
    #     inactives = load_protein_class(dataset, binders='inactive')

    # tf_idf_active, _ = identify_highlights(actives, vocabulary)
    # tf_idf_active.reset_index(inplace=True)
    # tf_idf_active.rename(columns={"index": "doc_id"}, inplace=True)
    # tf_idf_active = tf_idf_active.set_index("doc_id").T

    # tf_idf_inactive, _ = identify_highlights(inactives, vocabulary)
    # tf_idf_inactive.reset_index(inplace=True)
    # tf_idf_inactive.rename(columns={"index": "doc_id"}, inplace=True)
    # tf_idf_inactive = tf_idf_inactive.set_index("doc_id").T

    tf_idf_active = pd.read_csv(f'results/datasets/{dataset}/{vocabulary}/tf_idf_active.csv')
    tf_idf_inactive = pd.read_csv(f'results/datasets/{dataset}/{vocabulary}/tf_idf_inactive.csv')

    family_significance = []
    for col in set(tf_idf_active.columns) & set(tf_idf_inactive.columns):  
        family_df = pd.merge(tf_idf_active[col].reset_index(), tf_idf_inactive[col].reset_index(), on='index', how='outer')
        family_df = family_df.fillna(0)
        family_df.columns = ['word', 'active', 'inactive']
        # TODO: dropping words with no occurrence in particular family increases number of significant families
        family_df = family_df[(family_df['active'] != 0) | (family_df['inactive'] != 0)]
        _, upnorm = ranksums(family_df['active'].tolist(), family_df['inactive'].tolist())
        family_significance.append({'family': col, 'p-value': upnorm, 'significant': upnorm <= 0.05})
    output = pd.DataFrame(family_significance)
    if export:
        output_dir = Path(f'results/datasets/{dataset}/{vocabulary}')
        output_dir.mkdir(parents=True, exist_ok=True)
        output.to_csv(output_dir / 'significance.csv', index=False)
    return output

def find_longest_common_subsequence(word, vocab):
    nearest_neighbor = ''
    nearest_neighbor_len = 0
    word_len = compute_length(word)
    nearest_efg = ''
    for j, word2 in enumerate(vocab):
            res = pylcs.lcs_string_idx(word, word2)
            neighbor = ''.join([word2[i] for i in res if i != -1])
            neighbor_len = compute_length(neighbor)
            if neighbor_len > nearest_neighbor_len and neighbor_len <= word_len:
                nearest_neighbor = neighbor
                nearest_neighbor_len = compute_length(nearest_neighbor)
                nearest_efg = word2
            elif neighbor_len == nearest_neighbor_len:
                nearest_efg_len = compute_length(nearest_efg)
                efg_len = compute_length(word2)
                if efg_len < nearest_efg_len:
                    nearest_neighbor = neighbor
                    nearest_neighbor_len = compute_length(nearest_neighbor)
                    nearest_efg = word2

    return nearest_neighbor, nearest_efg

def find_nearest_neighbor(word, vocab):
    min_edit_distance = None
    nearest_efg = ''
    for j, word2 in enumerate(vocab):
            edit_dist = pylcs.edit_distance(word, word2)
            if min_edit_distance is None or edit_dist < min_edit_distance:
                min_edit_distance = edit_dist
                nearest_efg = word2
    return min_edit_distance, nearest_efg


def compute_jaccard_similarities(vocab1, vocab2):
    jaccard_similarities = np.zeros((len(vocab1), len(vocab2)))

    for i, word1 in tqdm(enumerate(vocab1)):
        for j, word2 in enumerate(vocab2):
            intersection = set(frag1).intersection(set(frag2))
            union = set(frag1).union(set(frag2))
            jaccard_similarities[i][j] = len(intersection) / len(union)

    # Compute the minimum Jaccard similarity for each fragment
    max_similarities = []
    for i, frag1 in enumerate(vocab1):
        similarities = []
        for j, frag2 in enumerate(vocab2):
            similarities.append(jaccard_similarities[i][j])
        max_similarities.append(max(similarities))
    return max_similarities


def compute_highlight_stats(dataset, vocabulary, plot=False): 
    if dataset == 'lit_pcba':
        documents, doc_to_name, doc_to_count = load_lit_pcba(binders='active', return_mappings=True)
    elif dataset == 'bdb':
        documents, doc_to_name, doc_to_count = load_bdb(binders='active', return_mappings=True)
    else:
        documents, doc_to_name, doc_to_count = load_protein_class(dataset, binders='active', return_mappings=True)

    tf_idf, highlighted_words = identify_highlights(documents, vocabulary, return_highlights=True)

    highlighted_words['word'] = highlighted_words[highlighted_words.columns[:]].apply(
        lambda x: [i for i in ','.join(x.dropna().astype(str)).split(',') if len(i) > 0],
        axis=1
    )
    # highlighted_words = pd.read_excel(f'results/datasets/{dataset}/{vocabulary}/active/highlighted_words.xlsx')
    highlighted_words_freq = highlighted_words.explode('word')['word'].reset_index().groupby('word').count()

    if plot: 
        output_dir = Path('figures/datasets') / dataset / vocabulary 
        output_dir.mkdir(parents=True, exist_ok=True)

        plt.figure()
        g = sns.histplot(highlighted_words_freq['index'], stat='percent', discrete=True)
        plt.savefig(output_dir / 'highlight_freq.png')
    else:
        return highlighted_words_freq

def compute_length(word):
    return len([token for token in regex.findall(word)])

def conduct_vocab_analysis(df_words):
    ''' Conducts a vocabulary analysis on a given vocabulary including the following: 
        -  Length of each word
        -  Longest common subsequence
        -  Length of the longest common subsequence
        -  Validity of each word'''
    
    # Load EFG vocabulary
    with open('data/chembl27_efg.json')  as f:
        efg_vocab = json.loads(f.read())['vocab']
    # Remove the special tokens
    df_words['word'] = df_words['word'].progress_apply(lambda w: w.replace('##', ''))
    # Check if the word is chemically valid
    df_words['valid'] = df_words['word'].progress_apply(lambda w: Chem.MolFromSmiles(w) is not None)
    # Compute the length of each word
    df_words['length'] = df_words['word'].progress_apply(compute_length)
    # Compute the longest common subsequence
    df_words['lcs'], df_words['efg'] = zip(*df_words['word'].progress_apply(lambda w: find_longest_common_subsequence(w, efg_vocab)))
    # Compute the length of the longest common subsequence
    df_words['lcs_length'] = df_words['lcs'].progress_apply(compute_length)
    # Compute nearest neighbor
    df_words['nearest_neighbor_dist'], df_words['nearest_efg'] = zip(*df_words['word'].progress_apply(lambda w: find_nearest_neighbor(w, efg_vocab)))
    df_words = df_words.assign(model=model, size=size)
    return df_words


def compute_coverage(dataset, tokenizer):
    ''' Computes the coverage of a given vocabulary on a given dataset. '''
    if dataset == 'lit_pcba':
        doc_active = pd.DataFrame(sum(load_lit_pcba(binders='active').values(), []), columns=['smiles'])
        doc_inactive = pd.DataFrame(sum(load_lit_pcba(binders='inactive').values(), []), columns=['smiles'])
    elif dataset == 'bdb':
        doc_active = pd.DataFrame(sum(load_bdb(binders='active').values(), []), columns=['smiles'])
        doc_inactive = pd.DataFrame(sum(load_bdb(binders='inactive').values(), []), columns=['smiles'])
    else:
        doc_active = pd.DataFrame(sum(load_protein_class(dataset, binders='active').values(), []), columns=['smiles'])
        doc_inactive = pd.DataFrame(sum(load_protein_class(dataset, binders='inactive').values(), []), columns=['smiles'])

    doc_active_filtered = doc_active.drop_duplicates()
    doc_inactive_filtered = doc_inactive.drop_duplicates()
    docs = pd.concat([doc_active_filtered.assign(binders='active'), doc_inactive_filtered.assign(binders='inactive')])
    docs['words'] = docs['smiles'].apply(lambda s: tokenizer.encode(s).tokens)
    docs_all_words = docs.explode('words').drop_duplicates('words')
    docs_active_words = docs[docs['binders'] == 'active'].explode('words').drop_duplicates('words')
    docs_inactive_words = docs[docs['binders'] == 'inactive'].explode('words').drop_duplicates('words')
    return {'overall': len(docs_all_words), 'active': len(docs_active_words), 'inactive': len(docs_inactive_words)}


if __name__ == '__main__': 
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--vocabulary', type=str)
    parser.add_argument('--vocab_size', type=str)
    args = parser.parse_args()
    if args.dataset is None:
        datasets = dataset_list
    else: 
        datasets = [args.dataset]
    if args.vocabulary is None:
        vocab_list = [p for p in Path('data/vocabs/').iterdir()]
    else:
        vocab_list = [args.vocabulary]
    tokenizers = {p.stem: Tokenizer.from_file(str(p)) for p in vocab_list}
    vocabs = {p: t.get_vocab() # [w.replace('##', '') for w in t.get_vocab()] 
               for p, t in tokenizers.items()}

    outputs = []
    highlight_stats = []
    vocab_intersections = {}
    coverage = []
    highlighted_words = {}
    for vocab_name, vocab in vocabs.items():
        vocab_intersections[vocab_name] = {}

        if args.vocab_size and args.vocab_size not in vocab:
            continue

        _, model, size = vocab_name.split('_')
        df_words = pd.DataFrame(vocab, columns=['word'])
        df_words = conduct_vocab_analysis(df_words)
        output_dir = Path('results/vocabs')  / vocab_name
        output_dir.mkdir(parents=True, exist_ok=True)
        df_words.to_csv(output_dir / 'vocab_analysis.csv', index=False)

        for vocab2_name, vocab2 in vocabs.items():
            if  vocab_name.split('_')[1] == vocab2_name.split('_')[1]:
                vocab_intersections[vocab_name][vocab2_name] =  len(set(vocabs[vocab_name]) & set(vocabs[vocab2_name])) 
            elif 'wordpiece' in vocab_name: 
                len1 = len(set([w for w in vocabs[vocab_name] if not w.startswith('##')]) & set(vocabs[vocab2_name]))
                len2 = len(set([w.replace('##', '') for w in vocabs[vocab_name] if w.startswith('##')]) & set(vocabs[vocab2_name]))
                vocab_intersections[vocab_name][vocab2_name] = len1 + len2
                print(len1, len2)
            elif 'wordpiece' in vocab2_name: 
                len1 = len(set([w for w in vocabs[vocab2_name] if not w.startswith('##')]) & set(vocabs[vocab_name]))
                len2 = len(set([w.replace('##', '') for w in vocabs[vocab2_name] if w.startswith('##')]) & set(vocabs[vocab_name]))
                vocab_intersections[vocab_name][vocab2_name] = len1 + len2
                print(len1, len2)

            else:
                vocab_intersections[vocab_name][vocab2_name] =  len(set(vocabs[vocab_name]) & set(vocabs[vocab2_name])) 

            print(f'Computing intersection between {vocab_name} and {vocab2_name}')
            print(len(set([w.replace('##', '') for w in vocabs[vocab_name] if not w.startswith('##')])))
            print(len(set([w.replace('##', '') for w in vocabs[vocab2_name] if not w.startswith('##')])))
            print(len(set([w.replace('##', '') for w in vocabs[vocab_name] if not w.startswith('##')])& set([w.replace('##', '') for w in vocabs[vocab2_name] if not w.startswith('##')])))
            print(len(set([w.replace('##', '') for w in vocabs[vocab_name] if w.startswith('##')])))
            print(len(set([w.replace('##', '') for w in vocabs[vocab2_name] if w.startswith('##')])))
            print(len(set([w.replace('##', '') for w in vocabs[vocab_name] if  w.startswith('##')]) & set([w.replace('##', '') for w in vocabs[vocab2_name] if w.startswith('##')])))

        highlighted_words[vocab_name] = {}
        for dataset in dataset_list:
            print(f'Conducting analysis for {dataset} using {vocab_name}')

            coverage.append({'dataset': dataset, 'vocab': vocab_name, **compute_coverage(dataset, tokenizers[vocab_name])})

            highlight_freqs = compute_highlight_stats(dataset, vocab_name)
            highlighted_words[vocab_name][dataset] = [w.replace('##', '') for w in highlight_freqs["word"].tolist()]
            highlight_stats.append({'dataset': dataset, 'vocab': vocab_name, 'num_highlight': highlight_freqs.shape[0]})

            output_dir = Path('results/datasets') / dataset / vocab_name
            output_dir.mkdir(parents=True, exist_ok=True)
            highlight_freqs.to_csv(output_dir / 'highlight_freqs.csv', index=False)

            plt.figure()
            g = sns.histplot(highlight_freqs['index'], stat='percent', discrete=True)
            plt.savefig(output_dir / 'highlight_freq.png')

            output = conduct_significance_analysis(dataset, vocab_name)
            outputs.append(output.assign(dataset=dataset, vocab=vocab_name))
    dataset_specific_highlights = []
    for dataset in dataset_list:
        for vocab_name, vocab in vocabs.items():
            for vocab2_name, vocab2 in vocabs.items():
                common_highlights = len(set(highlighted_words[vocab_name][dataset]) & set(highlighted_words[vocab2_name][dataset]))
                dataset_specific_highlights.append({'vocab1': vocab_name, 'vocab2': vocab2_name, 'dataset': dataset, 'common_highlights': common_highlights})
    
    outputs_all = pd.concat(outputs)
    outputs_summary = outputs_all.groupby(['dataset', 'vocab', 'significant'])['family'].count().reset_index()
    outputs_summary.to_csv('results/significance_summary.csv', index=False)
    outputs_all.to_csv('results/significance_overall.csv', index=False)
    pd.DataFrame(highlight_stats).to_csv('results/highlight_stats.csv', index=False)
    pd.DataFrame(vocab_intersections).to_csv('results/vocab_intersections.csv')
    pd.DataFrame(coverage).to_csv('results/coverage.csv', index=False)
    pd.DataFrame(dataset_specific_highlights).to_csv('results/dataset_specific_highlights.csv')


