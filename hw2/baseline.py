from utils import *
#from nltk.stem.snowball import SnowballStemmer
import kenlm

"""
from gensim.models import Word2Vec as W2vec
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Merge
from keras.layers.embeddings import Embedding
from keras.layers import LSTM
from keras.regularizers import l2
from keras.optimizers import SGD
"""
HYP_FILE = "./data/train-test.hyp1-hyp2-ref_tok_lower"
GOLD_FILE = "./data/train.gold"
OUTPUT_FILE = "./output.txt"
VEC_FILE = "./vecs/GoogleNews-vectors-negative300.bin"
TRADEOFF_PARAM = 0.4# between 0 and 1 ---> closer to 0 means recall heavy, else, precision heavy
#english_stemmer = SnowballStemmer("english")
"""
TRAIN_SPLIT_RATIO = 0.85
SENTENCE_EMBEDDING_LENGTH = 500
"""

def get_english_stem(word):
    global  english_stemmer
    english_stemmer.stem(word.strip().decode('utf-8')).encode('utf-8')

def get_word_count_dict(sentence):
    #word_list = [get_english_stem(word) for word in sentence.split(" ")]
    word_list = sentence.split(" ")
    tot_len = len(word_list)
    wcount_dict = {}
    for word in word_list:
        wcount_dict[word] = wcount_dict.get(word, 0.0) + 1.0
    return(wcount_dict, tot_len)

def get_precision(hyp, ref):
    """
    #Currently ignoring multiple occurrences of words
    ref_words = set(ref.split(" "))
    hyp_words = hyp.split(" ")
    wcount_hyp = len(hyp_words)
    match_count = 0.0
    for hyp_word in hyp_words:
        if hyp_word in ref_words:
            match_count += 1.0
    return(match_count/wcount_hyp)
    """
    # Accounting for number of occurrences of words
    [ref_wc, ref_len] = get_word_count_dict(ref)
    [hyp_wc, hyp_len] = get_word_count_dict(hyp)
    match_count = 0.0
    for hyp_word in hyp_wc.keys():
        if ref_wc.get(hyp_word, None) is not None:
            match_count += min(ref_wc[hyp_word], hyp_wc[hyp_word])
    return(match_count/hyp_len)


def get_recall(hyp, ref):
    """
    hyp_words = set(hyp.split(" "))
    ref_words = ref.split(" ")
    wcount_ref = len(ref_words)
    match_count = 0.0
    for ref_word in ref_words:
        if ref_word in hyp_words:
            match_count += 1.0
    return(match_count/wcount_ref)
    """
    # Accounting for number of occurrences of words
    [ref_wc, ref_len] = get_word_count_dict(ref)
    [hyp_wc, hyp_len] = get_word_count_dict(hyp)
    match_count = 0.0
    for ref_word in ref_wc.keys():
        if hyp_wc.get(ref_word, None) is not None:
            match_count += min(ref_wc[ref_word], hyp_wc[ref_word])
    return (match_count / ref_len)

def get_whm(prec, rec, tradeoff_param):
    if prec == 0 or rec == 0:
        return(-1)
    return(1.0 / ((tradeoff_param/prec) + ((1 - tradeoff_param)/rec)))



def get_prediction(parallel_instance, tradeoff_param):
    [hyp1, hyp2, ref] = parallel_instance
    h1_prec = get_precision(hyp1, ref)
    h1_rec = get_recall(hyp1, ref)
    h1_met = get_whm(h1_prec, h1_rec, tradeoff_param)
    h2_prec = get_precision(hyp2, ref)
    h2_rec = get_recall(hyp2, ref)
    h2_met = get_whm(h2_prec, h2_rec, tradeoff_param)

    if h1_met > h2_met:
        return(-1)
    elif h2_met > h1_met:
        return(1)
    else:
        return(0)

"""
print("Fetching google word vectors")

pretrained_w2vec = W2vec.load_word2vec_format(VEC_FILE, binary=True)  # C binary format

# Actually, try training your own word2vec model and use your vectors if they are absent from the google vecs
[labeled_instances, unlabeled_instances, w2vec_model, word2index, index2word] = get_data_and_w2vec_dicts(HYP_FILE, GOLD_FILE, pretrained_w2vec)

split_index = int(len(labeled_instances) * TRAIN_SPLIT_RATIO)
"""

#[labeled_instances, unlabeled_instances, w2vec_model, word2index, index2word] = get_data_and_w2vec_dicts(HYP_FILE, GOLD_FILE, pretrained_w2vec)
[labeled_instances, unlabeled_instances, word2index, index2word] = get_data_and_w2vec_dicts(HYP_FILE, GOLD_FILE)

model = kenlm.LanguageModel('europarl_en_lm.klm')
print(model.score('the european union'))

with open(OUTPUT_FILE, 'w') as op_file:
    #METEOR baseline method
    for instance in labeled_instances:
        parallel_instance = instance[0]
        op_file.write(str(get_prediction(parallel_instance, TRADEOFF_PARAM)) + '\n')
    for instance in unlabeled_instances:
        parallel_instance = instance
        op_file.write(str(get_prediction(parallel_instance, TRADEOFF_PARAM)) + '\n')
    print("Finished writing output file with meteor method")


"""
    [train_ref_tensor, train_hyp1_tensor, train_hyp2_tensor, train_common_3way_labels], \
    [test_ref_tensor, test_hyp1_tensor, test_hyp2_tensor, test_common_3way_labels], \
    max_sentence_len = get_matrized_data(labeled_instances, word2index, TRAIN_SPLIT_RATIO)

    print(labeled_instances[0][0][0])
    print(labeled_instances[0][0][1])
    print(labeled_instances[0][0][2])

    print(train_hyp1_tensor[0])
    print(train_hyp2_tensor[0])
    print(train_ref_tensor[0])

    print(labeled_instances[split_index][0][0])
    print(labeled_instances[split_index][0][1])
    print(labeled_instances[split_index][0][2])
    print(test_hyp1_tensor[0])
    print(test_hyp2_tensor[0])
    print(test_ref_tensor[0])


init_weights1 = np.random.rand(get_wvec_size(pretrained_w2vec), SENTENCE_EMBEDDING_LENGTH)
init_weights2 = np.random.rand(SENTENCE_EMBEDDING_LENGTH, SENTENCE_EMBEDDING_LENGTH)
init_weights3 = np.random.rand(SENTENCE_EMBEDDING_LENGTH,)

# Constructing the ref_model
print("Building ref_model")
ref_model = Sequential()
ref_model.add(Embedding(input_dim=len(word2index), output_dim=get_wvec_size(pretrained_w2vec), W_regularizer=l2(0.01), mask_zero=True, weights=[w2vec_model], dropout=0.2))
# don't specify init since we are seeding with weights. Input len is used when you have a sequential network on top. It
# is the max_length of a sentence in a matrix
print("Initialized embedding matrix with pretrained word vecs")
ref_model.add(LSTM(output_dim=SENTENCE_EMBEDDING_LENGTH, init='zero'))# weights=[init_weights1, init_weights2, init_weights3]))
ref_model.add(Activation('tanh'))
print("Added LSTM model")
"""

"""
# Single representation check
ref_model.add(Dense(output_dim=3, init='lecun_uniform', input_dim=SENTENCE_EMBEDDING_LENGTH))
ref_model.add(Activation('softmax'))
ref_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
ref_model.fit(train_hyp1_tensor, train_common_3way_labels, batch_size=1024, nb_epoch=30, validation_split=0.1, show_accuracy=True, verbose=1)
"""

"""
print("Building hyp1_model")
hyp1_model = Sequential()
hyp1_model.add(Embedding(input_dim=len(word2index), output_dim=get_wvec_size(pretrained_w2vec), W_regularizer=l2(0.01), mask_zero=True, weights=[w2vec_model], dropout=0.2))
# don't specify init since we are seeding with weights. Input len is used when you have a sequential network on top. It
# is the max_length of a sentence in a matrix
print("Initialized embedding matrix with pretrained word vecs")
hyp1_model.add(LSTM(output_dim=SENTENCE_EMBEDDING_LENGTH, init='zero')) #weights=[init_weights1, init_weights2, init_weights3]))
hyp1_model.add(Activation('tanh'))
print("Added LSTM model")

print("Building hyp1_model")
hyp2_model = Sequential()
hyp2_model.add(Embedding(input_dim=len(word2index), output_dim=get_wvec_size(pretrained_w2vec), W_regularizer=l2(0.01), mask_zero=True, weights=[w2vec_model], dropout=0.2))
# don't specify init since we are seeding with weights. Input len is used when you have a sequential network on top. It
# is the max_length of a sentence in a matrix
print("Initialized embedding matrix with pretrained word vecs")
hyp2_model.add(LSTM(output_dim=SENTENCE_EMBEDDING_LENGTH, init='zero')) #weights=[init_weights1, init_weights2, init_weights3]))
hyp2_model.add(Activation('tanh'))
print("Added LSTM model")


combined_model = Sequential()
combined_model.add(Merge([ref_model, hyp1_model, hyp2_model], mode='concat'))
#combined_model.add(Merge([ref_model, ref_model, ref_model], mode='concat'))

print("Added merge layer")

combined_model.add(Dense(output_dim=3, init='lecun_uniform', input_dim=3*SENTENCE_EMBEDDING_LENGTH))
combined_model.add(Activation('softmax'))
combined_model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.01, decay=0.0, nesterov=False)) #optimizer='rmsprop')
print("Added dense layer with softmax over classes")
print(train_ref_tensor.shape, train_hyp1_tensor.shape, train_hyp2_tensor.shape)
combined_model.fit([train_ref_tensor, train_hyp1_tensor, train_hyp2_tensor], train_common_3way_labels, batch_size=1024, nb_epoch=30, validation_split=0.1, show_accuracy=True, verbose=1)
"""