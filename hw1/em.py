# -*- coding: utf-8 -*-
from nltk.stem.snowball import SnowballStemmer
import random
import pickle
import itertools

"""
To do:
Package this into a class
Take in parameters as arguments
"""


N_LINES = 1000
INPUT_FILE = "./data/dev-test-train.de-en" # "./data/em_test.txt"
MAX_ITERS = 10


german_stemmer = SnowballStemmer("german")
english_stemmer = SnowballStemmer("english")

null_val = ''.decode('utf-8').encode('utf-8')

#Both vocabularies have a null
german_vocab = set([null_val])
english_vocab = set([null_val])

#Conditional probabilities are initialized to a uniform distribution over all german words conditioned on an english
#word. However, most words never co-occurr. To save space and avoid storing such parameters that are bound to be 0, we
#only store those conditional probabilities involving word pairs that actually co-occur.

translation_probs = {}

#Expected count number of alignments between each german word - english word pair

counts = {}

#Expected number of all alignments involving an english word (sum counts over all german words while fixing the english
#word

totals = {}


instance_preprocessed = 0
#may have to take care of unicode stuff. Also, you should probably split the compounded nouns apart.
#may have to append null to each sentence
def preprocess(line):
    global german_vocab, english_vocab, totals
    global instance_preprocessed
    if instance_preprocessed % 1000 == 0:
        print("Preprocessed {0} parallel inputs".format(instance_preprocessed))
    instance_preprocessed += 1
    [german, english] = line.strip().lower().split('|||')
    for word in german.split(' '):
        german_stemmed_word = german_stemmer.stem(word.strip().decode('utf-8')).encode('utf-8')
        german_vocab.add(german_stemmed_word)
    for word in english.split(' '):
        english_stemmed_word = english_stemmer.stem(word.strip().decode('utf-8')).encode('utf-8')
        english_vocab.add(english_stemmed_word)
        for word in german.split(' '):
            german_stemmed_word = german_stemmer.stem(word.strip().decode('utf-8')).encode('utf-8')
            key = (german_stemmed_word, english_stemmed_word)
            if not translation_probs.has_key(key):
                totals[english_stemmed_word] = totals.get(english_stemmed_word, 0) + 1.0
            translation_probs[key] = 1.0

def normalize():
    global english_vocab, german_vocab, totals, translation_probs, null_val
    for english_word in english_vocab:
        english_stemmed_word = english_stemmer.stem(english_word.decode('utf-8')).encode('utf-8')
        for german_word in german_vocab:
            german_stemmed_word = german_stemmer.stem(german_word.decode('utf-8')).encode('utf-8')
            key = (german_stemmed_word, english_stemmed_word)
            if translation_probs.has_key(key): #prevent populating entries unless they occur in parallel sentences
                translation_probs[key] = translation_probs[key] / totals[english_stemmed_word] #totals of english word should NEVER be 0
            elif english_stemmed_word == null_val:
                translation_probs[key] = 1.0 / len(german_vocab)

def get_parallel_instance(corpus_line):
    [german, english] = corpus_line.strip().lower().split('|||')
    ret_german = {}
    ret_english = {}
    for word in german.split(' '):
        german_stemmed_word = german_stemmer.stem(word.strip().decode('utf-8')).encode('utf-8')
        ret_german[german_stemmed_word] = ret_german.get(german_stemmed_word, 0) + 1
    for word in english.split(' '):
        english_stemmed_word = english_stemmer.stem(word.strip().decode('utf-8')).encode('utf-8')
        ret_english[english_stemmed_word] = ret_english.get(english_stemmed_word, 0) + 1
    ret_german[null_val] = 1 # null
    ret_english[null_val] = 1 # null
    return ([ret_german, ret_english])

ip_line_counter = 0

#May have to take care of last line being empty
with open(INPUT_FILE) as ip:
    print("Starting to process corpus")
    for line in ip:
        preprocess(line)
        ip_line_counter += 1
        if(ip_line_counter % 1000 == 0):
            print("Processed %d lines"%(ip_line_counter))

normalize()

print("No. of german words (stems) : " + str(len(german_vocab)))
print("No. of english words (stems) :" + str(len(english_vocab)))

iter_count = 0

def stop_condition(iter_count, max_change=MIN_CHANGE_THRESHOLD):
    if iter_count == MAX_ITERS or max_change < MIN_CHANGE_THRESHOLD:
        return(True)
    else:
        return(False)

while(True):#until convergence or max_iters
    print("Iteration " + str(iter_count + 1))
    iter_count += 1
    counts = {} # All counts default to 0. These are counts of (german, english) word pairs
    totals = {} # All totals default to 0. These are sums of counts (marginalized over all foreign words), for each
    # english word
    with open(INPUT_FILE) as ip_file: # Reading input one line at a time from file. No need to store file in memory
        for line in ip_file: # Read corpus in line by line instead of storing the whole thing in memory
            parallel_instance = get_parallel_instance(line) # returns two dicts mapping german and english words to
            # their respective counts in the parallel sentence pair
            german_sent_dict = parallel_instance[0]
            english_sent_dict = parallel_instance[1]
            for german_word in german_sent_dict.keys(): # For each unique german word in the german sentence
                german_word_count = german_sent_dict[german_word] # Its count in the german sentence
                total_s = 0.0  # Expected count of number of alignments for this german word with any english word
                for english_word in english_sent_dict.keys():
                    total_s += translation_probs.get((german_word, english_word), 0.0) * german_word_count
                for english_word in english_sent_dict.keys():
                    english_word_count  = english_sent_dict[english_word]
                    counts[(german_word, english_word)] = counts.get((german_word, english_word), 0.0) + translation_probs.get((german_word, english_word), 0.0) * german_word_count * english_word_count / total_s
                    # Expected count of alignments between german word and this english word, divided by the expected count of all alignments of this german word
                    totals[english_word] = totals.get(english_word, 0.0) + translation_probs.get((german_word, english_word), 0.0)* german_word_count * english_word_count / total_s
                    # Aggregating the expected counts of all german words, for each english word. This will be used as a normalizing factor
        for english_word in totals.keys(): # restricting to domain total( . )
            for word_pair in counts.keys():
                german_word = word_pair[0]
                if(word_pair[1] != english_word): # restricting to domain count( . | e )
                    continue
                translation_probs[word_pair] = counts.get((german_word, english_word), 0.0) / totals.get(english_word, 0.0)
                # Neither domain nor counts should never be 0 given our domain restriction

    if(iter_count % 2 == 0): #Store the model every other iteration
        print("Storing model after %d iterations"%(iter_count))
        model_dump = open('./translation.model', 'wb')
        pickle.dump(translation_probs, model_dump)
        model_dump.close()

    if(stop_condition(iter_count)):
        print("Storing model after %d iterations" % (iter_count))
        model_dump = open('./translation.model', 'wb')
        pickle.dump(translation_probs, model_dump)
        model_dump.close()
        break

test_english_words = [english_stemmer.stem(english_word.lower().strip(' \t\r\n').decode('utf-8')).encode('utf-8') for english_word in random.sample(english_vocab, 5)]
#Randomly select 5 words to perform sanity spot checks

print("Spot checking for the following English words ")
print(test_english_words)

for english_word in test_english_words:
    max_prob_word = None
    max_prob = 0.0
    tot_conditional_prob = 0.0
    for german_word in german_vocab:
        if translation_probs.get((german_word,english_word), 0.0) != 0.0:
            if translation_probs.get((german_word,english_word), 0.0) > max_prob:
                max_prob = translation_probs.get((german_word,english_word), 0.0)
                max_prob_word = german_word
            print("Probability p(%s | %s)"%(german_word, english_word))
            print(translation_probs.get((german_word,english_word), 0.0))
        tot_conditional_prob += translation_probs.get((german_word,english_word), 0.0)
    print('Tot_conditional prob = ' + str(tot_conditional_prob))
    print("Most likely word for ", english_word, " is the german word ", max_prob_word, " with translation probability ", max_prob)
    assert abs(tot_conditional_prob - 1.0) < 0.000000000001, 'Tot conditional probability != 1 !!!' # Difference may arise
    # due to finite precision, rounding or lack of representative ability of arbitrary reals using binary


print("Memory usage stats")
print("German vocab length: ", len(german_vocab))
print("English vocab length: ", len(english_vocab))
print("No of cross product entries required: ", len(german_vocab) * len(english_vocab))

print("Num of conditional probabilities actually stored: ", len(translation_probs.keys()))
print("Num of counts actually stored: ", len(counts.keys()))
print("Num of totals actually stored: ", len(totals.keys()))

