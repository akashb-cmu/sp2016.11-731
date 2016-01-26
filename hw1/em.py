from nltk.stem.snowball import SnowballStemmer
import pickle
import itertools
INPUT_FILE = "./data/em_test.txt"
MAX_ITERS = 10
MIN_CHANGE_THRESHOLD = 0.000001

german_stemmer = SnowballStemmer("german")
english_stemmer = SnowballStemmer("english")

#Both vocabularies have a null
german_vocab = set([''.decode('utf-8')])
english_vocab = set([''.decode('utf-8')])

#may have to take care of unicode stuff. Also, you should probably split the compounded nouns apart.
#may have to append null to each sentence
def preprocess(line):
    global german_vocab, english_vocab
    [german, english] = line.strip().lower().split('|||')
    ret_german = {}
    ret_english = {}
    for word in german.split(' '):
        german_stemmed_word = german_stemmer.stem(word.strip().decode('utf-8'))
        german_vocab.add(german_stemmed_word)
        ret_german[german_stemmed_word] = ret_german.get(german_stemmed_word, 0) + 1
    for word in english.split(' '):
        english_stemmed_word = english_stemmer.stem(word.decode('utf-8'))
        english_vocab.add(english_stemmed_word)
        ret_english[english_stemmed_word] = ret_english.get(english_stemmed_word, 0) + 1
    ret_german[''.decode('utf-8')] #null
    ret_english[''.decode('utf-8')] #null
    return([ret_german, ret_english])


ip = open(INPUT_FILE)

#May have to take care of last line being empty
corpus = [preprocess(line) for line in ip.readlines()]

ip.close()

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

print("No. of german words (stems) : " + str(len(german_vocab)))
print("No. of english words (stems) :" + str(len(english_vocab)))

iter_count = 0
translation_init_prob = 1.0/(len(german_vocab))

def stop_condition(iter_count, max_change):
    if iter_count == MAX_ITERS or max_change < MIN_CHANGE_THRESHOLD:
        return(True)
    else:
        return(False)

while(True):
    print("Iteration " + str(iter_count))
    iter_count += 1
    max_change = 0
    for parallel_instance in corpus:
        german_sent_dict = parallel_instance[0]
        english_sent_dict = parallel_instance[1]
        for german_word in german_sent_dict.keys():
            german_word_count = german_sent_dict[german_word]
            total_s = 0.0
            for english_word in english_sent_dict.keys():
                total_s += translation_probs.get((german_word, english_word), translation_init_prob) * german_word_count
            for english_word in english_sent_dict.keys():
                english_word_count  = english_sent_dict[english_word]
                counts[(german_word, english_word)] = counts.get((german_word, english_word), 0.0) + translation_probs.get((german_word, english_word), translation_init_prob) * german_word_count * english_word_count / total_s
                totals[english_word] = totals.get(english_word, 0.0) + translation_probs.get((german_word, english_word), translation_init_prob)* german_word_count * english_word_count / total_s
    for english_word in totals.keys():
        for german_word in german_vocab:
            if totals.get(english_word, 0.0) > 0 and abs(translation_probs.get((german_word, english_word), translation_init_prob) -  (counts.get((german_word, english_word), 0.0) / totals.get(english_word, 0.0))) > max_change:
                max_change = abs(translation_probs.get((german_word, english_word), translation_init_prob) -  counts.get((german_word, english_word), 0.0) / totals.get(english_word, 0.0))
            translation_probs[(german_word, english_word)] = counts.get((german_word, english_word), 0.0) / totals.get(english_word, 0.0)

    if(stop_condition(iter_count, max_change)):
        break

#test_english_words = [english_stemmer.stem(english_word.decode('utf-8')) for english_word in ['gentlemen', 'president', 'ladies', ',', '']]
test_english_words = []
for english_word in test_english_words:
    tot_conditional_prob = 0.0
    for german_word in german_vocab:
        print("Probability p(%s | %s)"%(german_word, english_word))
        print(translation_probs.get((german_word.decode('utf-8'),english_word), translation_init_prob))
        tot_conditional_prob += translation_probs.get((german_word.decode('utf-8'),english_word), translation_init_prob)
    print('Tot_conditional prob = ' + str(tot_conditional_prob))
    assert tot_conditional_prob != 1.0, 'Tot conditional probability != 1 !!!'


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