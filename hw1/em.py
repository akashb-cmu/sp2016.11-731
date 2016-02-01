from nltk.stem.snowball import SnowballStemmer
import pickle
import sys

INPUT_FILE = "./data/em_test.txt"
MAX_ITERS = 10
MIN_CHANGE_THRESHOLD = 0.000001

german_stemmer = SnowballStemmer("german")
english_stemmer = SnowballStemmer("english")

#Both vocabularies have a null
german_vocab = set([''.decode('utf-8')])
english_vocab = set([''.decode('utf-8')])

german_indexer = {''.decode('utf-8') : 0}
english_indexer = {''.decode('utf-8') : 0}
german_inv_indexer = {0 : ''.decode('utf-8')}
english_inv_indexer = {0 : ''.decode('utf-8')}

max_english_index = 1
max_german_index = 1


#vocab_cartesian = set(itertools.product(german_vocab, english_vocab))
#Each element is {<german word>, <english word> : 1/|german vocab|} which corresponds to p(german word | english word)
#Conditional probabilities are initialized to a uniform distribution over all german words conditioned on an english
#word. However, the initialization doesn't matter since EM for model 1 IBM is guaranteed to converge to the same values.
#Indeed this is why IBM models 2 - 5 build on top of model 1.
#translation_probs = dict.fromkeys(vocab_cartesian, 1.0/(len(german_vocab)))

translation_probs = {}

#Expected count number of alignments between each german word - english word pair
#counts = dict.fromkeys(vocab_cartesian, 0.0)

counts = {}

#Expected number of all alignments involving an english word (sum counts over all german words while fixing the english
#word
#totals = dict.fromkeys(english_vocab, 0.0)

totals = {}

#may have to take care of unicode stuff. Also, you should probably split the compounded nouns apart.
#may have to append null to each sentence
def preprocess(line):
    global german_vocab, english_vocab, german_indexer, german_inv_indexer, english_indexer, english_inv_indexer, max_german_index, max_english_index
    [german, english] = line.strip().lower().split('|||')
    for word in german.split(' '):
        german_stemmed_word = german_stemmer.stem(word.strip().decode('utf-8'))
        if (not german_indexer.has_key(german_stemmed_word)):
            german_indexer[german_stemmed_word] = max_german_index
            german_inv_indexer[max_german_index] = german_stemmed_word
            max_german_index += 1
            german_vocab.add(german_stemmed_word)
    for word in english.split(' '):
        english_stemmed_word = english_stemmer.stem(word.decode('utf-8'))
        if (not english_indexer.has_key(english_stemmed_word)):
            english_indexer[english_stemmed_word] = max_english_index
            english_inv_indexer[max_english_index] = english_stemmed_word
            max_english_index += 1
            english_vocab.add(english_stemmed_word)
        for word in german.split(' '):
            german_stemmed_word = german_stemmer.stem(word.strip().decode('utf-8'))
            key = (german_stemmed_word, english_stemmed_word)
            if not translation_probs.has_key(key):
                totals[english_stemmed_word] = totals.get(english_stemmed_word, 0) + 1.0
            translation_probs[key] = 1.0

def normalize():
    global english_vocab, german_vocab, totals, translation_probs
    null_val = ''.decode('utf-8')
    for english_word in english_vocab:
        english_stemmed_word = english_stemmer.stem(english_word.decode('utf-8'))
        for german_word in german_vocab:
            german_stemmed_word = german_stemmer.stem(german_word.decode('utf-8'))
            """
            print(german_word)
            try:
                german_stemmed_word = german_stemmer.stem(german_word.decode('utf-8'))
            except:
                e = sys.exc_info()[0]
                print("Failed because of ", e)
            """
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
        german_stemmed_word = german_stemmer.stem(word.strip().decode('utf-8'))
        ret_german[german_stemmed_word] = ret_german.get(german_stemmed_word, 0) + 1
    for word in english.split(' '):
        english_stemmed_word = english_stemmer.stem(word.decode('utf-8'))
        ret_english[english_stemmed_word] = ret_english.get(english_stemmed_word, 0) + 1
    ret_german[''.decode('utf-8')] = 1 # null
    ret_english[''.decode('utf-8')] = 1 # null
    return ([ret_german, ret_english])

#May have to take care of last line being empty
with open(INPUT_FILE) as ip:
    for line in ip:
        preprocess(line)

normalize()

print("No. of german words (stems) : " + str(len(german_vocab)))
print("No. of english words (stems) :" + str(len(english_vocab)))

iter_count = 0

def stop_condition(iter_count): # Currently only checking for iteration limit. Ideally, we should also check for
# convergence, i.e., when parameters change by value below a certain threshold
    if iter_count == MAX_ITERS:
        return(True)
    else:
        return(False)

"""
EM algorithm for estimating the translation probablities
See https://www.cl.cam.ac.uk/teaching/1011/L102/clark-lecture3.pdf for a good tutorial
"""

while(True):#until convergence or max_iters
    if iter_count % 5 == 0:
        print("Iteration " + str(iter_count))
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

    if(stop_condition(iter_count)):
        break

test_english_words = [english_stemmer.stem(english_word.lower().decode('utf-8')) for english_word in ['Deputy', 'gentlemen', 'president', 'welcome']]
#test_english_words = []
for english_word in test_english_words:
    tot_conditional_prob = 0.0
    for german_word in german_vocab:
        print("Probability p(%s | %s)"%(german_word, english_word))
        print(translation_probs.get((german_word.decode('utf-8'),english_word), 0.0))
        tot_conditional_prob += translation_probs.get((german_word.decode('utf-8'),english_word), 0.0)
    print('Tot_conditional prob = ' + str(tot_conditional_prob))
    assert tot_conditional_prob - 1.0 < 0.000000000001, 'Tot conditional probability != 1 !!!' # Difference may arise
    # due to finite precision, rounding or lack of representative ability of arbitrary reals using binary


print("Memory usage stats")
print("German vocab length: ", len(german_vocab))
print("English vocab length: ", len(english_vocab))
print("No of cross product entries required: ", len(german_vocab) * len(english_vocab))

print("Num of conditional probabilities actually stored: ", len(translation_probs.keys()))
print("Num of counts actually stored: ", len(counts.keys()))
print("Num of totals actually stored: ", len(totals.keys()))
#MAKE A CHECK HERE FOR SUMMATION OF PROBABILITIES TO ONE!!

model_dump = open('./translation.model', 'wb')
pickle.dump(translation_probs, model_dump)
model_dump.close()