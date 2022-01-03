"""
nlp, assignment 4, 2021

In this assignment you will implement a Hidden Markov model and an LSTM model
to predict the part of speech sequence for a given sentence.
(Adapted from Nathan Schneider)

"""

import torch
import torch.nn as nn
import torchtext as tt
from torchtext import data
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import pandas as pd
from math import log, isfinite
from collections import Counter
import numpy as np
import sys, os, time, platform, nltk, random

# With this line you don't need to worry about the HW  -- GPU or CPU
# GPU cuda cores will be used if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# You can call use_seed with other seeds or None (for complete randomization)
# but DO NOT change the default value.
def use_seed(seed=2512021):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_deterministic(True)
    # torch.backends.cudnn.deterministic = True


# utility functions to read the corpus
def who_am_i():  # this is not a class method
    """Returns a dictionary with your name, id number and email. keys=['name', 'id','email']
        Make sure you return your own info!
    """
    return {'name': 'Michael Michaelshvili', 'id': '318949443', 'email': 'michmich@post.bgu.ac.il'}


def read_annotated_sentence(f):
    line = f.readline()
    if not line:
        return None
    sentence = []
    while line and (line != "\n"):
        line = line.strip()
        word, tag = line.split("\t", 2)
        sentence.append((word, tag))
        line = f.readline()
    return sentence


def load_annotated_corpus(filename):
    sentences = []
    with open(filename, 'r', encoding='utf-8') as f:
        sentence = read_annotated_sentence(f)
        while sentence:
            sentences.append(sentence)
            sentence = read_annotated_sentence(f)
    return sentences


START = "<DUMMY_START_TAG>"
END = "<DUMMY_END_TAG>"
UNK = "<UNKNOWN>"
PAD = "<PAD>"

allTagCounts = Counter()
# use Counters inside these
perWordTagCounts = {}
transitionCounts = {}
emissionCounts = {}
# log probability distributions: do NOT use Counters inside these because
# missing Counter entries default to 0, not log(0)
A = {}  # transisions probabilities
B = {}  # emmissions probabilities


def fill_count_dict(dict, k1, k2):
    if k1 not in dict:
        dict[k1] = Counter({k2: 1})
    else:
        dict[k1][k2] += 1


def get_distribution_probabilities(from_dict):
    probs = {}
    for k1 in from_dict:
        sum_sounts = sum(from_dict[k1].values())
        probs[k1] = {k2: log(c / sum_sounts) for k2, c in from_dict[k1].items()}
    return probs


def learn_params(tagged_sentences):
    """Populates and returns the allTagCounts, perWordTagCounts, transitionCounts,
    and emissionCounts data-structures.
    allTagCounts and perWordTagCounts should be used for baseline tagging and
    should not include pseudocounts, dummy tags and unknowns.
    The transisionCounts and emmisionCounts
    should be computed with pseudo tags and shoud be smoothed.
    A and B should be the log-probability of the normalized counts, based on
    transisionCounts and  emmisionCounts

    Args:
    tagged_sentences: a list of tagged sentences, each tagged sentence is a
     list of pairs (w,t), as retunred by load_annotated_corpus().

    Return:
    [allTagCounts,perWordTagCounts,transitionCounts,emissionCounts,A,B] (a list)
    """
    for sen in tagged_sentences:
        prev_state = START
        for word, tag in sen:
            allTagCounts[tag] += 1
            fill_count_dict(perWordTagCounts, word, tag)
            fill_count_dict(transitionCounts, prev_state, tag)
            fill_count_dict(emissionCounts, tag, word)
            prev_state = tag
        fill_count_dict(transitionCounts, prev_state, END)

    # smoothing
    for key1 in transitionCounts.keys():
        c = transitionCounts[key1]
        for key2 in list(allTagCounts.keys()) + [END]:
            c[key2] += 1
    del transitionCounts[START][END]
    for key1 in emissionCounts.keys():
        c = emissionCounts[key1]
        for key2 in list(perWordTagCounts.keys()) + [UNK]:
            c[key2] += 1

    A = get_distribution_probabilities(transitionCounts)
    B = get_distribution_probabilities(emissionCounts)

    return [allTagCounts, perWordTagCounts, transitionCounts, emissionCounts, A, B]


def baseline_tag_sentence(sentence, perWordTagCounts, allTagCounts):
    """Returns a list of pairs (w,t) where each w corresponds to a word
    (same index) in the input sentence. Each word is tagged by the tag most
    frequently associated with it. OOV words are tagged by sampling from the
    distribution of all tags.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        perWordTagCounts (Counter): tags per word as specified in learn_params()
        allTagCounts (Counter): tag counts, as specified in learn_params()

        Return:
        list: list of pairs
    """
    tagged_sentence = []
    for word in sentence:
        if word in perWordTagCounts:
            tagged_sentence.append((word, perWordTagCounts[word].most_common(1)[0][0]))
        else:
            sampled_tag = random.choices(list(allTagCounts.keys()), weights=list(allTagCounts.values()), k=1)
            tagged_sentence.append((word, sampled_tag))
    return tagged_sentence


# ===========================================
#       POS tagging with HMM
# ===========================================


def hmm_tag_sentence(sentence, A, B):
    """Returns a list of pairs (w,t) where each w corresponds to a word
    (same index) in the input sentence. Tagging is done with the Viterby
    algorithm.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        A (dict): The HMM Transition probabilities
        B (dict): tthe HMM emmission probabilities.

    Return:
        list: list of pairs
    """

    tags = retrace(viterbi(sentence, A, B))
    tagged_sentence = list(zip(sentence, tags))
    return tagged_sentence


def viterbi(sentence, A, B):
    """Creates the Viterbi matrix, column by column. Each column is a list of
    tuples representing cells. Each cell ("item") is a tupple (t,r,p), were
    t is the tag being scored at the current position,
    r is a reference to the corresponding best item from the previous position,
    and p is a log probabilityof the sequence so far).

    The function returns the END item, from which it is possible to
    trace back to the beginning of the sentence.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        A (dict): The HMM Transition probabilities
        B (dict): tthe HMM emmission probabilities.

    Return:
        obj: the last item, tagged with END. should allow backtraking.

        """
    # Hint 1: For efficiency reasons - for words seen in training there is no
    #      need to consider all tags in the tagset, but only tags seen with that
    #      word. For OOV you have to consider all tags.
    # Hint 2: start with a dummy item  with the START tag (what would it log-prob be?).
    #         current list = [ the dummy item ]
    # Hint 3: end the sequence with a dummy: the highest-scoring item with the tag END
    sentence.insert(0, START)
    viterbi = pd.DataFrame(0, index=list(allTagCounts.keys()), columns=[i for i in range(len(sentence))])
    for t in A[START]:
        viterbi.loc[t, 0] = A[START][t] + (B[t][sentence[0]] if sentence[0] in perWordTagCounts else B[t][UNK])

    for w in range(1, len(sentence)):  # loop through observations (words)
        tags_to_iterate = list(perWordTagCounts[sentence[w]].keys()) if sentence[w] in perWordTagCounts else list(
            allTagCounts.keys())
        for t in tags_to_iterate:  # loop through the tags
            best_tag, best_prob = predict_next_best(sentence, w, t, viterbi, A, B)
            viterbi.loc[t, w] = best_prob
    v_last = {'tag': END, 'p': None}
    curr_v = v_last
    for i in range(len(sentence) - 1, -1, -1):
        if i == 0:  # START
            curr_v['p'] = {'tag': START, 'p': None}
        else:
            curr_v['p'] = {'tag': viterbi.iloc[:, i][viterbi.iloc[:, i] < 0].idxmax(), 'p': None}
            curr_v = curr_v['p']
    sentence.pop(0)
    return v_last


# a suggestion for a helper function. Not an API requirement
def retrace(end_item):
    """Returns a list of tags (retracing the sequence with the highest probability,
        reversing it and returning the list). The list should correspond to the
        list of words in the sentence (same indices).
    """
    tags = []
    item = end_item['p']
    while item['p']:
        tags.append(item['tag'])
        item = item['p']
    return reversed(tags)


# a suggestion for a helper function. Not an API requirement
def predict_next_best(sentence, word_idx, tag, viterbi, A, B):
    """Returns a new item (tupple)
    """
    best_tag, best_prob = None, float('-inf')

    for s in allTagCounts:
        prob = viterbi.at[s, word_idx - 1] + A[s][tag] + (
            B[tag][sentence[word_idx]] if sentence[word_idx] in B[tag] else B[tag][UNK])
        if prob > best_prob:
            best_prob = prob
            best_tag = s
    return best_tag, best_prob


def joint_prob(sentence, A, B):
    """Returns the joint probability of the given sequence of words and tags under
     the HMM model.

     Args:
         sentence (pair): a sequence of pairs (w,t) to compute.
         A (dict): The HMM Transition probabilities
         B (dict): the HMM emmission probabilities.
     """
    p = 0  # joint log prob. of words and tags

    prev_tag = START
    for word, tag in sentence:
        p += A[prev_tag][tag] + (B[tag][word] if word in B[tag] else B[tag][UNK])
        prev_tag = tag

    p += A[prev_tag][END]

    assert isfinite(p) and p < 0  # Should be negative. Think why!
    return p


# ===========================================
#       POS tagging with BiLSTM
# ===========================================

""" You are required to support two types of bi-LSTM:
    1. a vanilla biLSTM in which the input layer is based on simple word embeddings
    2. a case-based BiLSTM in which input vectors combine a 3-dim binary vector
        encoding case information, see
        https://arxiv.org/pdf/1510.06168.pdf
"""


# Suggestions and tips, not part of the required API
#
#  1. You can use PyTorch torch.nn module to define your LSTM, see:
#     https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM
#  2. You can have the BLSTM tagger model(s) implemented in a dedicated class
#     (this could be a subclass of torch.nn.Module)
#  3. Think about padding.
#  4. Consider using dropout layers
#  5. Think about the way you implement the input representation
#  6. Consider using different unit types (LSTM, GRU,LeRU)


def initialize_rnn_model(params_d):
    """Returns a dictionary with the objects and parameters needed to run/train_rnn
       the lstm model. The LSTM is initialized based on the specified parameters.
       thr returned dict is may have other or additional fields.

    Args:
        params_d (dict): a dictionary of parameters specifying the model. The dict
                        should include (at least) the following keys:
                        {'max_vocab_size': max vocabulary size (int),
                        'min_frequency': the occurence threshold to consider (int),
                        'input_rep': 0 for the vanilla and 1 for the case-base (int),
                        'embedding_dimension': embedding vectors size (int),
                        'num_of_layers': number of layers (int),
                        'output_dimension': number of tags in tagset (int),
                        'pretrained_embeddings_fn': str,
                        'data_fn': str
                        }
                        max_vocab_size sets a constraints on the vocab dimention.
                            If the its value is smaller than the number of unique
                            tokens in data_fn, the words to consider are the most
                            frequent words. If max_vocab_size = -1, all words
                            occuring more that min_frequency are considered.
                        min_frequency privides a threshold under which words are
                            not considered at all. (If min_frequency=1 all words
                            up to max_vocab_size are considered;
                            If min_frequency=3, we only consider words that appear
                            at least three times.)
                        input_rep (int): sets the input representation. Values:
                            0 (vanilla), 1 (case-base);
                            <other int>: other models, if you are playful
                        The dictionary can include other keys, if you use them,
                             BUT you shouldn't assume they will be specified by
                             the user, so you should spacify default values.
    Return:
        a dictionary with the at least the following key-value pairs:
                                       {'lstm': torch.nn.Module object,
                                       input_rep: [0|1]}
        #Hint: you may consider adding the embeddings and the vocabulary
        #to the returned dict
    """

    data_fn = params_d['data_fn']
    sentences = load_annotated_corpus(data_fn)

    counter = Counter(w.lower() for sentence in sentences for w, t in sentence)
    max_vocab_size = params_d['max_vocab_size']
    min_frequency = params_d['min_frequency']
    vocab = []
    word_frequency_list = counter.most_common(max_vocab_size if max_vocab_size != -1 else None)
    for word, f in word_frequency_list:
        if f >= min_frequency:
            vocab.append(word)

    vectors = load_pretrained_embeddings(params_d['pretrained_embeddings_fn'], vocab)
    max_input_size = max([len(sentence) for sentence in sentences])
    all_tags_from_data = {t for sentence in sentences for w, t in sentence} | {END}
    ttoi, itot = {}, {}
    for i, t in enumerate(all_tags_from_data):
        ttoi[t] = i
        itot[i] = t

    input_rep = params_d['input_rep']
    lstm = RNN(max_input_size=max_input_size, embedding_dim=params_d['embedding_dimension'],
               pretrained_weights=vectors.vectors, input_rep=input_rep,
               num_of_layers=params_d['num_of_layers'], output_dimension=params_d['output_dimension'] + 1)

    model = {'lstm': lstm, 'input_rep': input_rep, 'wtoi': vectors.stoi, 'itow': vectors.itos, 'ttoi': ttoi,
             'itot': itot}
    return model

    # no need for this one as part of the API
    # def get_model_params(model):
    """Returns a dictionary specifying the parameters of the specified model.
    This dictionary should be used to create another instance of the model.

    Args:
        model (torch.nn.Module): the network architecture

    Return:
        a dictionary, containing at least the following keys:
        {'input_dimension': int,
        'embedding_dimension': int,
        'num_of_layers': int,
        'output_dimension': int}
    """

    # TODO complete the code

    # return params_d


def get_word_feaure(word, wtoi, input_rep):
    idx = wtoi[PAD] if word == PAD else (wtoi[word.lower()] if word.lower() in wtoi else wtoi[UNK])
    case = []
    if input_rep == 1:
        case = [word.islower(), word.isupper(), word[0].isupper()]
    return idx, case


def get_sentence_features(words, wtoi, input_rep):
    idxs, cases = [], []
    for word in words:
        idx, case = get_word_feaure(word, wtoi, input_rep)
        idxs.append(idx)
        cases.append(case)
    return idxs, cases


def get_X_y_from_sentences(sentences, wtoi, ttoi, max_length, input_rep):
    X_train_idx, x_train_case, y_train = [], [], []
    for sentence in sentences:
        tag_idx = []  # y - the value to predict
        words = []
        for word, tag in sentence:
            tag_idx.append(ttoi[tag])
            words.append(word)
        words += (max_length - len(words)) * [PAD]
        tag_idx += (max_length - len(tag_idx)) * [ttoi[END]]
        y_train.append(tag_idx)
        idxs, cases = get_sentence_features(words, wtoi, input_rep)
        X_train_idx.append(idxs)
        x_train_case.append(cases)
    X_train_idx = torch.LongTensor(X_train_idx)
    x_train_case = torch.BoolTensor(x_train_case)
    y_train = torch.LongTensor(y_train)
    return X_train_idx, x_train_case, y_train


def load_pretrained_embeddings(path, vocab=None):
    """ Returns an object with the the pretrained vectors, loaded from the
        file at the specified path. The file format is the same as
        https://www.kaggle.com/danielwillgeorge/glove6b100dtxt
        You can also access the vectors at:
         https://www.dropbox.com/s/qxak38ybjom696y/glove.6B.100d.txt?dl=0
         (for efficiency (time and memory) - load only the vectors you need)
        The format of the vectors object is not specified as it will be used
        internaly in your code, so you can use the datastructure of your choice.

    Args:
        path (str): full path to the embeddings file
        vocab (list): a list of words to have embeddings for. Defaults to None.

    """

    class Vector:
        def __init__(self, itos, stoi, vectors):
            self.itos = itos
            self.stoi = stoi
            self.vectors = vectors

    vectors = tt.vocab.Vectors(path)
    if vocab is not None:
        dict = {PAD: 0, UNK: 1}
        dict.update({k: i + 2 for i, k in enumerate(vocab)})
        vectors = Vector([PAD, UNK] + vocab, dict,
                         torch.cat([torch.Tensor([[0.0] * 100]), torch.Tensor([[0.0] * 100]),
                                    vectors.get_vecs_by_tokens(vocab)]))
    else:
        dict = {PAD: 0, UNK: 1}
        dict.update({k: v + 2 for k, v in vectors.stoi.items()})
        vectors = Vector([PAD, UNK] + vectors.itos, dict,
                         torch.cat([torch.Tensor([[0.0] * 100]), torch.Tensor([[0.0] * 100]), vectors.vectors]))
    return vectors


class RNN(nn.Module):
    def __init__(self, max_input_size, embedding_dim, pretrained_weights, input_rep,
                 num_of_layers, output_dimension, hidden_dim=128, dropout=0.25):
        super().__init__()
        self.input_rep = input_rep
        self.max_input_size = max_input_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding.from_pretrained(pretrained_weights, freeze=True)
        self.lstm = nn.LSTM(embedding_dim + 3 * input_rep, hidden_dim, num_layers=num_of_layers, dropout=dropout,
                            bidirectional=True)
        self.fc = nn.Linear(2 * hidden_dim, output_dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        emb_idx = input[0]
        case_features = input[1]
        x = self.dropout(self.embedding(emb_idx))
        if self.input_rep == 1:
            x = torch.cat((x, case_features), dim=2)
        x, (h, c) = self.lstm(x)
        x = self.fc(self.dropout(x))
        return x

    def fit(self, x_train, y_train, opt, criterion, x_val=None, y_val=None, epochs=20, batch_size=128):
        train_ds = TensorDataset(x_train[0], x_train[1], y_train)
        train_dl = DataLoader(train_ds, batch_size=batch_size)
        best_loss = None
        for epoch in range(epochs):
            self.train()
            for xb1, xb2, yb in train_dl:
                opt.zero_grad()
                y_pred = self([xb1.to(device), xb2.to(device)])
                y_pred = y_pred.reshape(-1, y_pred.shape[-1])
                yb = yb.reshape(-1)
                loss = criterion(y_pred, yb.to(device))
                loss.backward()
                opt.step()
            if x_val:
                self.eval()
                y_pred = self([x_val[0].to(device), x_val[1].to(device)])
                y_pred = y_pred.reshape(-1, y_pred.shape[-1])
                y_val = y_val.reshape(-1).to(device)
                valid_loss = criterion(y_pred, y_val)
                if best_loss is None or valid_loss < best_loss:
                    best_loss = valid_loss
                    torch.save(self.state_dict(), f'best_model_rep_{self.input_rep}.pt')

                print(f'epoch: {epoch}\t loss: {valid_loss}')


def train_rnn(model, train_data, val_data=None):
    """Trains the BiLSTM model on the specified data.

    Args:
        model (dict): the model dict as returned by initialize_rnn_model()
        train_data (list): a list of annotated sentences in the format returned
                            by load_annotated_corpus()
        val_data (list): a list of annotated sentences in the format returned
                            by load_annotated_corpus() to be used for validation.
                            Defaults to None
        input_rep (int): sets the input representation. Defaults to 0 (vanilla),
                         1: case-base; <other int>: other models, if you are playful
    """
    # Tips:
    # 1. you have to specify an optimizer
    # 2. you have to specify the loss function and the stopping criteria
    # 3. consider using batching
    # 4. some of the above could be implemented in helper functions (not part of
    #    the required API)
    model, input_rep, wtoi, itow, ttoi, itot = model.values()
    opt = optim.Adam(model.parameters(), lr=0.001)

    criterion = nn.CrossEntropyLoss()  # you can set the parameters as you like

    model = model.to(device)
    criterion = criterion.to(device)

    x_train_idx, x_train_case, y_train = get_X_y_from_sentences(train_data, wtoi, ttoi, model.max_input_size, input_rep)
    if val_data:
        X_val_idx, x_val_case, y_val = get_X_y_from_sentences(val_data, wtoi, ttoi, model.max_input_size, input_rep)
    else:
        idxs = np.arange(len(x_train_idx))
        np.random.shuffle(idxs)
        train_idx, val_idx = idxs[int(0.2 * len(idxs)):], idxs[:int(0.2 * len(idxs))]

        X_val_idx, x_val_case, y_val = x_train_idx[val_idx], x_train_case[val_idx], torch.LongTensor(y_train[val_idx])
        x_train_idx, x_train_case, y_train = x_train_idx[train_idx], x_train_case[train_idx], torch.LongTensor(
            y_train[train_idx])
    model.fit([x_train_idx, x_train_case], y_train, opt, criterion, x_val=[X_val_idx, x_val_case], y_val=y_val)
    model.load_state_dict(torch.load(f'best_model_rep_{input_rep}.pt'))


def rnn_tag_sentence(sentence, model):
    """ Returns a list of pairs (w,t) where each w corresponds to a word
        (same index) in the input sentence and t is the predicted tag.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        model (dict):  a dictionary with the trained BiLSTM model and all that is needed
                        to tag a sentence.

    Return:
        list: list of pairs
    """
    model, input_rep, wtoi, itow, ttoi, itot = model.values()
    pad_sentence = sentence + (model.max_input_size - len(sentence)) * [PAD]
    idxs, cases = get_sentence_features(pad_sentence, wtoi, input_rep)
    idxs, cases = torch.LongTensor(idxs).reshape(1, -1).to(device), torch.BoolTensor(cases).reshape(1, -1, 3).to(device)
    pred = model([idxs, cases])
    indices = torch.argmax(pred[:len(sentence)], dim=-1)[0]
    indices = [int(i) for i in indices]
    tags = [itot[idx] if idx in itot else UNK for idx in indices]
    tagged_sentence = list(zip(sentence, tags))
    return tagged_sentence


def get_best_performing_model_params():
    """Returns a disctionary specifying the parameters of your best performing
        BiLSTM model.
        IMPORTANT: this is a *hard coded* dictionary that will be used to create
        a model and train a model by calling
               initialize_rnn_model() and train_lstm()
    """

    return {'max_vocab_size': -1,
            'min_frequency': 2,
            'input_rep': 1,
            'embedding_dimension': 100,
            'num_of_layers': 3,
            'output_dimension': 17,
            'pretrained_embeddings_fn': 'glove.6B.100d.txt',
            'data_fn': 'en-ud-train.upos.tsv'
            }


# ===========================================================
#       Wrapper function (tagging with a specified model)
# ===========================================================

def tag_sentence(sentence, model):
    """Returns a list of pairs (w,t) where pair corresponds to a word (same index) in
    the input sentence. Tagging is done with the specified model.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        model (dict): a dictionary where key is the model name and the value is
           an ordered list of the parameters of the trained model (baseline, HMM)
           or the model isteld and the input_rep flag (LSTMs).

        Models that must be supported (you can add more):
        1. baseline: {'baseline': [perWordTagCounts, allTagCounts]}
        2. HMM: {'hmm': [A,B]}
        3. Vanilla BiLSTM: {'blstm':[model_dict]}
        4. BiLSTM+case: {'cblstm': [model_dict]}
        5. (NOT REQUIRED: you can add other variations, agumenting the input
            with further subword information, with character-level word embedding etc.)

        The parameters for the baseline model are:
        perWordTagCounts (Counter): tags per word as specified in learn_params()
        allTagCounts (Counter): tag counts, as specified in learn_params()

        The parameters for the HMM are:
        A (dict): The HMM Transition probabilities
        B (dict): tthe HMM emmission probabilities.

        Parameters for an LSTM: the model dictionary (allows tagging the given sentence)


    Return:
        list: list of pairs
    """
    if list(model.keys())[0] == 'baseline':
        return baseline_tag_sentence(sentence, model['baseline'][0], model['baseline'][1])
    if list(model.keys())[0] == 'hmm':
        return hmm_tag_sentence(sentence, model['hmm'][0], model['hmm'][1])
    if list(model.keys())[0] == 'blstm':
        return rnn_tag_sentence(sentence, model['blstm'][0])
    if list(model.keys())[0] == 'cblstm':
        return rnn_tag_sentence(sentence, model['cblstm'][0])


def count_correct(gold_sentence, pred_sentence):
    """Return the total number of correctly predicted tags,the total number of
    correcttly predicted tags for oov words and the number of oov words in the
    given sentence.

    Args:
        gold_sentence (list): list of pairs, assume to be gold labels
        pred_sentence (list): list of pairs, tags are predicted by tagger

    """
    assert len(gold_sentence) == len(pred_sentence)
    correct, correctOOV, OOV = 0, 0, 0
    for gold_w, pred_s in zip(gold_sentence, pred_sentence):
        is_correct = gold_w[1] == pred_s[1]
        is_oov = gold_w[0] in perWordTagCounts
        if is_correct:
            correct += 1
        if is_oov:
            OOV += 1
        if is_correct and is_oov:
            correctOOV += 1

    return correct, correctOOV, OOV


if __name__ == '__main__':
    vectors = load_pretrained_embeddings('glove.6B.100d.txt')
    x = 0
