from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram, WordPiece
from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordPieceTrainer
from tokenizers.pre_tokenizers import WhitespaceSplit
from argparse import ArgumentParser
# os.environ["TOKENIZERS_PARALLELISM"] = "false"


class WordIdentifier:
    @classmethod
    def from_file(cls, path):
        instance = cls()
        instance.tokenizer = Tokenizer.from_file(path)
        instance.tokenizer.add_tokens(["[UNK]"])
        return instance

    def train_word_identifier_model(self, model_type, corpus_path, save_name, vocab_size, starting_vocab):
        if model_type not in {'bpe', 'unigram'}:
            raise ValueError('Unknown word identification algorithm')

        print(f'WordIdentifier: Training a model with vocab {vocab_size} on {corpus_path}')

        if model_type == 'bpe':
            tokenizer = Tokenizer(BPE())
            trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=['[PAD]'] + starting_vocab)
        elif model_type == 'unigram':
            tokenizer = Tokenizer(Unigram())
            trainer = UnigramTrainer(vocab_size=vocab_size, special_tokens=['[PAD]'] + starting_vocab)
        elif model_type == 'wordpiece': 
            tokenizer = Tokenizer(WordPiece())
            trainer = WordPieceTrainer(vocab_size=vocab_size, special_tokens=['[PAD]'] + starting_vocab)

        tokenizer.pre_tokenizer = WhitespaceSplit()
        tokenizer.train([corpus_path], trainer)
        tokenizer.save(save_name, pretty=True)
        self.tokenizer = tokenizer

    def identify_words(self, sequences, padding_len=None, out_type='int'):
        encodings = self.tokenizer.encode_batch(sequences)
        if padding_len is not None:
            for encoding in encodings:
                encoding.pad(padding_len, direction='right', pad_id=0, pad_token='[PAD]')
                encoding.truncate(padding_len)

        if out_type == 'int':
            return [encoding.ids for encoding in encodings]
        elif out_type == 'str':
            return [encoding.tokens for encoding in encodings]
        else:
            raise ValueError('Invalid out_type for word identification')



if __name__ == '__main__': 
    parser = ArgumentParser()
    parser.add_argument('--model_type', type=str, default='bpe', choices=['bpe', 'unigram', 'wordpiece'])
    parser.add_argument('--corpus', type=str, default='data/chembl27.txt')
    parser.add_argument('--save_name', type=str)
    parser.add_argument('--vocab_size', type=int, default=8000)
    args = parser.parse_args()
    identifier = WordIdentifier()
    # Warning: [UNK] is added only for WordPiece
    identifier.train_word_identifier_model(args.model_type, args.corpus, args.save_name, args.vocab_size, ["[UNK]"])