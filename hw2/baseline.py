from utils import *
#from nltk.stem.snowball import SnowballStemmer
import kenlm
import nltk
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import pickle
from nltk.corpus import stopwords

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
ERRORS_FILE = "./errors.txt"
TRADEOFF_PARAM = 0.45# between 0 and 1 ---> closer to 0 means recall heavy, else, precision heavy
#Best so far is 0.45
LM_CONTEXT_SIZE = 5
#FEATS = ['prec','rec','whm','lm_score', 'lda', 'len_word', 'len_char] #prec, rec and whm cannot be toggled off
FEATS = ['whm', 'len_word', 'len_char'] #prec, rec and whm cannot be toggled off
TRAIN_SPLIT = 0.9
EXTRA_FEAT_FILE = './ldafeatures.pickle'



#english_stemmer = SnowballStemmer("english")
"""
TRAIN_SPLIT_RATIO = 0.85
SENTENCE_EMBEDDING_LENGTH = 500
"""

def get_english_stem(word):
    global  english_stemmer
    english_stemmer.stem(word.strip().decode('utf-8')).encode('utf-8')

def get_word_count_dict(sentence):
    stop = set(stopwords.words('english'))
    #word_list = [get_english_stem(word) for word in sentence.split(" ")]
    """
    if len(sentence.split()) >= 5:
        word_list = [word for word in sentence.split(" ") if word.decode('utf8') not in stop]
    else:
        word_list = sentence.split(" ")
    """
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
    if hyp_len == 0:
        print(hyp)
        raw_input()
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


def get_prec_rec_whm(parallel_instance, tradeoff_param):
    [hyp1, hyp2, ref] = parallel_instance
    h1_prec = get_precision(hyp1, ref)
    h1_rec = get_recall(hyp1, ref)
    h1_met = get_whm(h1_prec, h1_rec, tradeoff_param)
    h2_prec = get_precision(hyp2, ref)
    h2_rec = get_recall(hyp2, ref)
    h2_met = get_whm(h2_prec, h2_rec, tradeoff_param)
    return([h1_prec, h1_rec, h1_met], [h2_prec, h2_rec, h2_met])



def get_prediction(parallel_instance, tradeoff_param):
    # [hyp1, hyp2, ref] = parallel_instance
    # h1_prec = get_precision(hyp1, ref)
    # h1_rec = get_recall(hyp1, ref)
    # h1_met = get_whm(h1_prec, h1_rec, tradeoff_param)
    # h2_prec = get_precision(hyp2, ref)
    # h2_rec = get_recall(hyp2, ref)
    # h2_met = get_whm(h2_prec, h2_rec, tradeoff_param)

    [h1_prec, h1_rec, h1_met], [h2_prec, h2_rec, h2_met] = get_prec_rec_whm(parallel_instance, tradeoff_param)

    if h1_met > h2_met:
        return(-1)
    elif h2_met > h1_met:
        return(1)
    else:
        return(0)

def get_feat_vects(feats, parallel_instance, lda_feats, tradeoff_param=0.5 , lm_context=3):
    global FEATS
    f_vect = []
    [h1_prec, h1_rec, h1_met], [h2_prec, h2_rec, h2_met] = get_prec_rec_whm(parallel_instance, tradeoff_param)
    if 'prec' in feats:
        f_vect.extend([h1_prec, h2_prec])
    if 'rec' in feats:
        f_vect.extend([h1_rec, h2_rec])
    if 'whm' in feats:
        f_vect.extend([h1_met, h2_met])
    #f_vect.extend([h1_prec, h2_prec, h1_rec, h2_rec, h1_met, h2_met])
    if 'lm_score' in feats:
        assert lm_context == 3 or lm_context == 5, "Invalid LM specified"
        if lm_context == 3:
            f_vect.append(trigram_lm.score(parallel_instance[0]))
            f_vect.append(trigram_lm.score(parallel_instance[1]))
            f_vect.append(trigram_lm.score(parallel_instance[2]))
            # f_vect.append(trigram_lm.score(parallel_instance[0])*1.0 / len(nltk.sent_tokenize(parallel_instance[0])))
            # f_vect.append(trigram_lm.score(parallel_instance[1]))
            # f_vect.append(trigram_lm.score(parallel_instance[2]))
        else:
            f_vect.append(pentagram_lm.score(parallel_instance[0]))
            f_vect.append(pentagram_lm.score(parallel_instance[1]))
            f_vect.append(pentagram_lm.score(parallel_instance[2]))
    if 'lda' in feats:
        f_vect.extend(lda_feats)
    if 'len_word' in feats:
        f_vect.extend([len(parallel_instance[0].split(" ")), len(parallel_instance[1].split(" ")), len(parallel_instance[2].split(" "))])
    if 'len_char' in feats:
        f_vect.extend([len(parallel_instance[0]), len(parallel_instance[1]), len(parallel_instance[2])])
    return(np.array(f_vect))

def get_accuracy(predicted_labels, gold_labels):
    match_count = 0
    for i in range(len(predicted_labels)):
        if predicted_labels[i] == gold_labels[i]:
            match_count += 1
    return(match_count * 1.0/len(gold_labels))


def get_wrong_instances(predictions, gold_labels, instances, error_file):
    assert len(predictions) == len(instances) and len(predictions) == len(gold_labels), "Predictions and instances don't line up"
    wrong_instances = []
    for index, pred in enumerate(predictions):
        if gold_labels[index] != pred:
            wrong_instances.append((instances[index][0][0], instances[index][0][1], instances[index][0][2]))
    with open(error_file, 'w') as error_f:
        for wrong_instance in wrong_instances:
            error_f.write(str(wrong_instance[0]) + '|||' + str(wrong_instance[1]) + '|||' + str(wrong_instance[2]) + '\n')




"""
print("Fetching google word vectors")

pretrained_w2vec = W2vec.load_word2vec_format(VEC_FILE, binary=True)  # C binary format

# Actually, try training your own word2vec trigram_lm and use your vectors if they are absent from the google vecs
[labeled_instances, unlabeled_instances, w2vec_model, word2index, index2word] = get_data_and_w2vec_dicts(HYP_FILE, GOLD_FILE, pretrained_w2vec)

split_index = int(len(labeled_instances) * TRAIN_SPLIT_RATIO)
"""

#[labeled_instances, unlabeled_instances, w2vec_model, word2index, index2word] = get_data_and_w2vec_dicts(HYP_FILE, GOLD_FILE, pretrained_w2vec)
[labeled_instances, unlabeled_instances, word2index, index2word] = get_data_and_w2vec_dicts(HYP_FILE, GOLD_FILE)

[labeled_lda_feats, unlabeled_lda_feats] = pickle.load(open(EXTRA_FEAT_FILE,'rb'))

assert len(labeled_lda_feats) == len(labeled_instances) and len(unlabeled_lda_feats) == len(unlabeled_instances), "Features and instances don't match!"

trigram_lm = kenlm.LanguageModel('europarl_en_lm_3gram.klm')
pentagram_lm = kenlm.LanguageModel('europarl_en_lm_5gram.klm')
print(trigram_lm.score('the european union'))
print(pentagram_lm.score('the european union'))

#train_feats = np.zeros((len(labeled_instances), 2*len(FEATS) + 1))
train_feats = []
#train_labels = np.zeros((len(labeled_instances),))
train_labels = []

#test_feats = np.zeros((len(unlabeled_instances), 2*len(FEATS) + 1))
test_feats = []

with open(OUTPUT_FILE, 'w') as op_file:
    #METEOR baseline method
    print("Assembling data matrices")
    for index, instance in enumerate(labeled_instances):
        parallel_instance = instance[0]
        label = instance[1]
        #op_file.write(str(get_prediction(parallel_instance, TRADEOFF_PARAM)) + '\n')
        #train_feats[index] = get_feat_vects(FEATS, parallel_instance, tradeoff_param=TRADEOFF_PARAM, lm_context=LM_CONTEXT_SIZE)
        train_feats.append(get_feat_vects(FEATS, parallel_instance, labeled_lda_feats[index], tradeoff_param=TRADEOFF_PARAM, lm_context=LM_CONTEXT_SIZE))
        #train_labels[index] = label
        train_labels.append(label)
    train_feats = np.array(train_feats)
    train_labels = np.array(train_labels)
    split_index = int(TRAIN_SPLIT * len(train_feats))
    validation_feats = train_feats[split_index:]
    validation_labels = train_labels[split_index:]
    train_feats = train_feats[:split_index]
    train_labels = train_labels[:split_index]

    print("Train dims:", train_feats.shape, train_labels.shape)
    print("Val dims:", validation_feats.shape, validation_labels.shape)

    for index,instance in enumerate(unlabeled_instances):
        parallel_instance = instance
        #op_file.write(str(get_prediction(parallel_instance, TRADEOFF_PARAM)) + '\n')
        #test_feats[index] = get_feat_vects(FEATS, parallel_instance, tradeoff_param=TRADEOFF_PARAM, lm_context=LM_CONTEXT_SIZE)
        test_feats.append(get_feat_vects(FEATS, parallel_instance, unlabeled_lda_feats[index], tradeoff_param=TRADEOFF_PARAM,
                                         lm_context=LM_CONTEXT_SIZE))

    test_feats = np.array(test_feats)
    print("Test dims:", test_feats.shape)

    """
    # Logistic Regression:
    model = LogisticRegression()
    model = model.fit(train_feats, train_labels)
    print(model.score(validation_feats, validation_labels))
    """

    print("Training SVM")

    #svm_model = svm.SVC(decision_function_shape='ovo')
    svm_model = svm.SVC()
    svm_model.fit(train_feats, train_labels)
    preds = svm_model.predict(np.concatenate((train_feats, validation_feats, test_feats)))
    #print(get_accuracy(preds, validation_labels))
    #dec = svm_model.decision_function(validation_feats)
    #print(dec)

    all_labels = np.concatenate((train_labels, validation_labels))

    get_wrong_instances(preds[:len(all_labels)], all_labels, labeled_instances, ERRORS_FILE)


    for pred in preds:
        op_file.write(str(pred) + '\n')
    print("Finished writing output file with meteor method")



"""
#LSTM approach
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
print("Added LSTM trigram_lm")
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
print("Added LSTM trigram_lm")

print("Building hyp1_model")
hyp2_model = Sequential()
hyp2_model.add(Embedding(input_dim=len(word2index), output_dim=get_wvec_size(pretrained_w2vec), W_regularizer=l2(0.01), mask_zero=True, weights=[w2vec_model], dropout=0.2))
# don't specify init since we are seeding with weights. Input len is used when you have a sequential network on top. It
# is the max_length of a sentence in a matrix
print("Initialized embedding matrix with pretrained word vecs")
hyp2_model.add(LSTM(output_dim=SENTENCE_EMBEDDING_LENGTH, init='zero')) #weights=[init_weights1, init_weights2, init_weights3]))
hyp2_model.add(Activation('tanh'))
print("Added LSTM trigram_lm")


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