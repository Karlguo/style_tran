# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import re
import unicodedata
import six
import tensorflow as tf
from collections import Counter
import json
import numpy as np
from bpe import Encoder
import random
from nltk.tokenize import SpaceTokenizer, WhitespaceTokenizer
import argparse

required_tokens = ["$number", "$person", "$date", "$time", "$term","$literal"]
special_terms = ['<pad>', '<unk>', '<sos>', '<eos>']
def printable_text(text):
    """Returns text encoded in a way suitable for print or `tf.logging`."""

    # These functions want `str` for both Python2 and Python3, but in one case
    # it's a Unicode string and in the other it's a byte string.
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text
        elif isinstance(text, unicode):
            return text.encode("utf-8")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")

def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def validate_case_matches_checkpoint(do_lower_case, init_checkpoint):
    """Checks whether the casing config is consistent with the checkpoint name."""

    # The casing has to be passed in by the user and there is no explicit check
    # as to whether it matches the checkpoint. The casing information probably
    # should have been stored in the bert_config.json file, but it's not, so
    # we have to heuristically detect it to validate.

    if not init_checkpoint:
        return

    m = re.match("^.*?([A-Za-z0-9_-]+)/bert_model.ckpt", init_checkpoint)
    if m is None:
        return

    model_name = m.group(1)

    lower_models = [
        "uncased_L-24_H-1024_A-16", "uncased_L-12_H-768_A-12",
        "multilingual_L-12_H-768_A-12", "chinese_L-12_H-768_A-12"
    ]

    cased_models = [
        "cased_L-12_H-768_A-12", "cased_L-24_H-1024_A-16",
        "multi_cased_L-12_H-768_A-12"
    ]

    is_bad_config = False
    if model_name in lower_models and not do_lower_case:
        is_bad_config = True
        actual_flag = "False"
        case_name = "lowercased"
        opposite_flag = "True"

    if model_name in cased_models and do_lower_case:
        is_bad_config = True
        actual_flag = "True"
        case_name = "cased"
        opposite_flag = "False"

    if is_bad_config:
        raise ValueError(
            "You passed in `--do_lower_case=%s` with `--init_checkpoint=%s`. "
            "However, `%s` seems to be a %s model, so you "
            "should pass in `--do_lower_case=%s` so that the fine-tuning matches "
            "how the model was pre-training. If this error is wrong, please "
            "just comment out this check." % (actual_flag, init_checkpoint,
                                              model_name, case_name, opposite_flag))



def create_vocab_bpe(data_path = 'data/mt_corpus_ts.txt', vocab_path='data/vocab.txt', vocab_size=25000,
                      simple_subword = False):
    print('start create vocab from:', data_path)
    line_num = 0
    encoder = Encoder(vocab_size, pct_bpe=0.75, ngram_min=1, ngram_max=4, required_tokens=required_tokens,
                      word_tokenizer=WhitespaceTokenizer().tokenize)
    texts = []
    with open(data_path, encoding='utf-8') as fin:
        for line in fin:
            line_num += 1
            if line_num == 1:
                continue
            if line_num % 100000 == 0:
                print(line_num)
            #tuples = line.strip().split('\t')
            #zh = tuples[1]
            texts.append( line )
    encoder.fit(texts)
    bpe_dict = encoder.vocabs_to_dict()
    with open(vocab_path+'.dict', 'w', encoding='utf-8') as fout:
        fout.write(json.dumps(bpe_dict))

    terms = list(encoder.bpe_vocab.keys())
    terms += list(encoder.word_vocab.keys())
    terms = set(terms)
    terms = list(terms)
    vocabs = special_terms + terms
    vocabs_dict = dict()
    for i, term in enumerate(vocabs):
        vocabs_dict[term] = i
    if not simple_subword:
        for i, term in enumerate(vocabs):
            if term not in special_terms and term not in required_tokens:
                vocabs_dict["@@"+term] = i+len(vocabs)
    with open(vocab_path, 'w', encoding='utf-8') as fout:
        fout.write(json.dumps(vocabs_dict, indent=0))
    print('create vocab done. save to: ', vocab_path)

def load_vocab_bpe(vocab_path):
    with open(vocab_path, encoding='utf-8') as fin:
        char2idx = json.loads(fin.read())
    idx2char = dict( (idx, char) for char, idx in char2idx.items() )
    return char2idx, idx2char

def remove_seow(terms, encoder, simple_subword):
    if simple_subword:
        terms = [term for term in terms if term != encoder.SOW and term != encoder.EOW]
        return terms
    new_terms = []
    in_word = False
    first_subword = True
    for term in terms:
        if term == encoder.SOW:
            in_word = True
            first_subword = True
        elif term == encoder.EOW:
            in_word = False
            first_subword = True
        elif in_word:
            if first_subword:
                new_terms.append(term)
                first_subword = False
            else:
                new_terms.append("@@" + term)
        else:
            if term == encoder.UNK:
                term = "<unk>"
            elif term == encoder.PAD:
                term = "<pad>"
            new_terms.append(term)
    return new_terms


class DataGenerator(object):
    def __init__(self, data_path="data/mt_corpus_ts.txt",
                   vocab_path = 'data/vocab_bpe.txt',
                 timesteps_max = 100,
                 batch_size = 256,
                 word_dropout_ratio=0.75, simple_subword=False):
        self.data_path = data_path
        self.simple_subword = simple_subword
        self.word_dropout_ratio = word_dropout_ratio
        self.timesteps_max = timesteps_max
        #load vocab
        self.char2idx, self.idx2char = load_vocab_bpe(vocab_path)
        with open(vocab_path + '.dict', encoding='utf-8') as fin:
            bpe_dict = json.loads(fin.read())
        self.encoder = Encoder.from_dict(bpe_dict)
        self.encoder.word_tokenizer = WhitespaceTokenizer().tokenize
        self.batch_size = batch_size

    def generator(self):
        rng = random.Random(88)
        # vectorize the data
        lines = []
        print('read data from ', self.data_path)
        line_num = 0
        num_encoder_tokens = max(self.idx2char.keys()) + 1
        print("Number of unique input tokens:", num_encoder_tokens)
        print("Max sequence length for inputs:", self.timesteps_max)
        while True:
            with open(self.data_path, encoding='utf-8') as fin:
                for line in fin:
                    line_num += 1
                    if line_num == 1:
                        continue
                    tuples = line.strip().split('\t')
                    zh = tuples[1]
                    terms = zh.split()
                    if len(terms) <= self.timesteps_max-2:
                        terms = self.encoder.tokenize(zh)
                        # terms = [term for term in terms if term != encoder.EOW and term != encoder.SOW]
                        terms = remove_seow(terms, self.encoder, self.simple_subword)
                        if len(terms) <= self.timesteps_max-2:
                            lines.append(terms)
                    if len(lines)== self.batch_size:
                        input_texts = []
                        for line in lines[: min(self.batch_size, len(lines))]:
                            input_text = line
                            input_text.append("<eos>")
                            input_texts.append(input_text)
                        #clean lines
                        lines = []
                        max_encoder_seq_length = self.timesteps_max
                        encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length), dtype="int32")
                        decoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length), dtype="int32")
                        decoder_output_data = np.zeros((len(input_texts), max_encoder_seq_length), dtype="int32")
                        for i, input_text in enumerate(input_texts):
                            decoder_input_data[i, 0] = self.char2idx["<sos>"]
                            for t, char in enumerate(input_text):
                                idx = self.char2idx[char] if char in self.char2idx else self.char2idx["<unk>"]
                                idx_mask = idx
                                if rng.random() < self.word_dropout_ratio:
                                    # TODO 添加一个新的单词<mask>，而不是使用<unk>
                                    if rng.random() < 0.9:
                                        idx_mask = self.char2idx["<unk>"]
                                    else:
                                        # 10% of the time, replace with random word
                                        idx_mask = rng.randint(0, num_encoder_tokens - 1)
                                encoder_input_data[i, t] = idx_mask
                                decoder_output_data[i, t] = idx
                                decoder_input_data[i, t + 1] = idx_mask
                        yield [encoder_input_data, decoder_input_data, np.expand_dims(decoder_output_data, axis=-1)], None



def read_text_data_bpe(num_samples=3000, data_path="data/mt_corpus_ts.txt",
                   vocab_path = 'data/vocab_bpe.txt', word_dropout_ratio=0.75,
                   simple_subword=False):
    """

    :param num_samples:
    :param data_path:
    :return: timesteps_max, char2id, id2char, x, x_decoder #enc_tokens, characters,
    """
    rng = random.Random(88)
    # vectorize the data
    timesteps_max = 100
    input_texts = []
    char2idx, idx2char = load_vocab_bpe(vocab_path)
    with open(vocab_path+'.dict', encoding='utf-8') as fin:
        bpe_dict = json.loads(fin.read())
    encoder = Encoder.from_dict(bpe_dict)
    encoder.word_tokenizer = WhitespaceTokenizer().tokenize
    lines = []
    print('read data from ', data_path)
    line_num = 0
    with open(data_path, encoding='utf-8') as fin:
        for line in fin:
            line_num += 1
            if line_num == 1:
                continue
            if line_num % 100000 == 0:
                print(line_num)
            if line_num > num_samples+200:
                break
            #tuples = line.strip().split('\t')
            #zh = tuples[1]
            zh = line
            terms = zh.split()
            if len(terms)<=timesteps_max-2:
                terms = encoder.tokenize(zh)
                # terms = [term for term in terms if term != encoder.EOW and term != encoder.SOW]
                terms = remove_seow(terms, encoder, simple_subword)
                if len(terms) <= timesteps_max-2:
                    lines.append(terms)

    for line in lines[: min(num_samples, len(lines) - 1)]:
        input_text = line
        input_text.append("<eos>")
        input_texts.append(input_text)
    #     for char in input_text:
    #         if char not in input_characters:
    #             input_characters.add(char)
    #
    # input_characters = sorted(list(input_characters))
    num_encoder_tokens = max(idx2char.keys())+1
    max_encoder_seq_length = timesteps_max#max([len(txt) for txt in input_texts]) + 1

    print("Number of samples:", len(input_texts))
    print("Number of unique input tokens:", num_encoder_tokens)
    print("Max sequence length for inputs:", max_encoder_seq_length)

    # input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
    # reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())

    encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length), dtype="int32")
    decoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length), dtype="int32")
    decoder_output_data = np.zeros((len(input_texts), max_encoder_seq_length), dtype="int32")
    # encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype="float32")
    # decoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype="float32")

    for i, input_text in enumerate(input_texts):
        decoder_input_data[i, 0] = char2idx["<sos>"]
        # decoder_input_data[i, 0, char2idx["<sos>"]] = 1.0
        for t, char in enumerate(input_text):
            idx = char2idx[char] if char in char2idx else char2idx["<unk>"]
            idx_mask = idx
            if rng.random() < word_dropout_ratio:
                #TODO 添加一个新的单词<mask>，而不是使用<unk>
                if rng.random()<0.9:
                    idx_mask = char2idx["<unk>"]
                else:
                    # 10% of the time, replace with random word
                    idx_mask = rng.randint(0, num_encoder_tokens - 1)
            encoder_input_data[i, t] = idx_mask
            decoder_output_data[i, t] = idx
            decoder_input_data[i, t + 1] = idx_mask
            # encoder_input_data[i, t, idx] = 1.0
            # decoder_input_data[i, t + 1, idx ] = 1.0

    return max_encoder_seq_length, char2idx, idx2char, encoder_input_data, decoder_input_data, decoder_output_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='建立字典')
    parser.add_argument('--data_path', type=str, help="数据路径",
                        default='/home/guandan.cgd/mt_corpus_finetune/mt_corpus_tn.txt')
    parser.add_argument('--vocab_path', type=str, help="vocab路径",
                        default='data/vocab_bpe.txt')
    parser.add_argument('--simple_subword', type=bool, help="是否不使用@@作为subword的开始",
                        default=False)
    parser.add_argument('--vocab_size', type=int, help="vocab size", default=40000)
    args = parser.parse_args()
    # create_vocab(data_path='/home/guandan.cgd/mt_corpus_finetune/mt_corpus_tn.txt')
    #'/home/guandan.cgd/mt_corpus_finetune/mt_corpus_tn.txt', 'data/vocab_bpe.txt'
    create_vocab_bpe(data_path=args.data_path, vocab_path=args.vocab_path,
                     simple_subword=args.simple_subword,
                     vocab_size=args.vocab_size)
    # create_vocab_bpe(data_path='data/mt_corpus_ts.txt', vocab_path='data/vocab_bpe.txt')
    # bpe_tokenize_test()
    # print('a  a'.split())



def convert_by_vocab(vocab, items):
    """Converts a sequence of [tokens|ids] using the vocab."""
    output = []
    for item in items:
        output.append(vocab[item])
    return output

def convert_by_vocab_clean(vocab, items):
    """Converts a sequence of [tokens|ids] using the vocab.
    自动过滤<sos> <eos>及其之后的文本
    """
    output = []
    for item in items:
        if vocab[item] == "<sos>":
            continue
        if vocab[item] == "<eos>":
            break
        output.append(vocab[item])
    return output

def convert_tokens_to_ids(vocab, tokens):
    return convert_by_vocab(vocab, tokens)

class FullTokenizer(object):
    """Runs end-to-end tokenziation."""

    def __init__(self, vocab_file, do_lower_case=True, simple_subword=False):
        self.vocab, self.inv_vocab = load_vocab_bpe(vocab_file)
        self.vocab_size = max(self.vocab.values())+1
        self.simple_subword = simple_subword
        with open(vocab_file + '.dict', encoding='utf-8') as fin:
            bpe_dict = json.loads(fin.read())
        self.encoder = Encoder.from_dict(bpe_dict)
        self.encoder.word_tokenizer = WhitespaceTokenizer().tokenize

    def get_vocab_size(self):
        return self.vocab_size

    def tokenize(self, text):
        terms = self.encoder.tokenize(text)
        # terms = [term for term in terms if term != encoder.EOW and term != encoder.SOW]
        terms = remove_seow(terms, self.encoder, self.simple_subword)
        return terms

    def convert_tokens_to_ids(self, tokens):
        return convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):
        return convert_by_vocab(self.inv_vocab, ids)

    def convert_ids_to_tokens_clean(self, ids):
        return convert_by_vocab_clean(self.inv_vocab, ids)

