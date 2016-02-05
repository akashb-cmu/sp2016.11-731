# -*- coding: utf-8 -*-
from nltk.stem.snowball import SnowballStemmer
import random
import pickle
import itertools
import pdb
#from aligntools import PosTagger

class Model1(object):
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

    def __init__(self, parameter_file):
        self.translation_probs = pickle.load(open(parameter_file,'rb'))
        self.INPUT_FILE = None

    def get_params(self):
        return(self.translation_probs)

    def get_german_stem(self, word):
        return(self.german_stemmer.stem(word.strip().decode('utf-8')).encode('utf-8'))

    def get_english_stem(self, word):
        return(self.english_stemmer.stem(word.strip().decode('utf-8')).encode('utf-8'))

    def get_translation_prob(self, german_stem, english_stem):
        return self.translation_probs.get((german_stem,english_stem),0.0)

    def get_parallel_instance(self, corpus_line):
        [german, english] = corpus_line.strip().lower().split(' ||| ')
        return ([self.german_stemmer.stem(word.strip().decode('utf-8')).encode('utf-8')
                 for word in german.split(' ')],
                [self.english_stemmer.stem(word.strip().decode('utf-8')).encode('utf-8')
                 for word in english.split(' ')])

    def get_counts(sent):
        """ 
        Returns a dicts mapping stemmed words to
        their respective counts in the sentence sent.
        Also, one null token count is added to sentence
        """
        ret = {self.null_val:1} 
        for word in sent:
            ret[word] = ret.get(word, 0) + 1
        return ret

    def get_prior(self,**kwargs):
        return 1

    def get_alignment(self, german, english):
        """
        Returns model1 alignment for a DE/EN parallel sentence pair.
        For each german word, identifies the best english word (or NULL) to align to
        """
        english.append(self.null_val)
        alignment = []
        for (i, g_i) in enumerate(german): 
            gs_i = self.get_german_stem(g_i)
            best = -1
            bestscore = 0
            for (j, e_j) in enumerate(english):
                es_j = self.get_english_stem(e_j)
                val = self.get_prior()*self.get_translation_prob(gs_i,es_j)
                if best==-1 or val>bestscore:
                    bestscore = val
                    best = j
            if best < len(english)-1:
                yield (i,best) # don't yield anything for NULL alignment

class POS_decoder(Model1):
    TUNE_POS_WEIGHT = 1

    def __init__(self, parameter_file):
        super(POS_decoder,self).__init__(parameter_file)
        self.tagger = PosTagger()

    def get_prior(self, **features):
        """
        returns 1+TUNE_POS_WEIGHT if the POS tags are aligned else 1
        """
        return 1+self.TUNE_POS_WEIGHT*(
            features.get(tag_german,self.null_val)==features.get(tag_english,self.null_val))

    def get_alignment(self, german, english):
        """
        Returns Model1 alignment for a DE/EN parallel sentence pair.
        For each german word, identifies the best english word (or NULL) to align to
        Applies a prior which assigns higher probability to alignments which preserve POS tags.
        """
        alignment = []
        gtags = self.tagger.parse(german,"DE")
        etags = self.tagger.parse(english,"EN")
        for (i, g_i) in enumerate(german): 
            gs_i = self.get_german_stem(g_i)
            best = -1
            bestscore = self.prior(tag_german=gtags[i])*self.get_translation_prob(gs_i,self.null_val)
            for (j, e_j) in enumerate(english):
                es_j = self.get_english_stem(e_j)
                val = self.get_prior(tag_german=gtags[i],tag_english=etags[j])*\
                      self.get_translation_prob(gs_i,es_j)
                if best==-1 or val>bestscore:
                    bestscore = val
                    best = j
            if best >= 0:
                yield (i,best) # don't yield anything for NULL alignment
        

class EM_model1(Model1):

    def __init__(self, input_file, output_file, n_iterations):
        self.MAX_ITERS = n_iterations
        self.INPUT_FILE = input_file
        self.OUTPUT_FILE = output_file

        ip_line_counter = 0

        # May have to take care of last line being empty
        with open(self.INPUT_FILE) as ip:
            print("Starting to process corpus")
            for line in ip:
                self.preprocess(line)
                ip_line_counter += 1
                if (ip_line_counter % 1000 == 0):
                    print("Processed %d lines" % (ip_line_counter))

        self.normalize()

        """
        #Pre-EM sanity check for initialization
        test_english_words = [english_word for english_word in english_vocab]
        #test_english_words = [english_stemmer.stem(english_word.lower().strip(' \t\r\n').decode('utf-8')).encode('utf-8') for english_word in ['please']]

        print("Spot checking before EM ")
        print(test_english_words)

        print("After normalization case:")

        for english_word in test_english_words:
            max_prob_word = None
            max_prob = 0.0
            tot_conditional_prob = 0.0
            for german_word in german_vocab:
                if translation_probs.get((german_word,english_word), 0.0) != 0.0:
                    if translation_probs.get((german_word,english_word), 0.0) > max_prob:
                        max_prob = translation_probs.get((german_word,english_word), 0.0)
                        max_prob_word = german_word
                tot_conditional_prob += translation_probs.get((german_word,english_word), 0.0)
            assert abs(tot_conditional_prob - 1.0) < 0.000000000001, 'Tot conditional probability != 1 !!!'
            print("Most likely word for ", english_word, " is the german word ", max_prob_word, " with translation probability ", max_prob)
        """

        print("No. of german words (stems) : " + str(len(self.german_vocab)))
        print("No. of english words (stems) :" + str(len(self.english_vocab)))



    #may have to take care of unicode stuff. Also, you should probably split the compounded nouns apart.
    #may have to append null to each sentence
    def preprocess(self, line):
        [german, english] = line.strip().lower().split('|||')
        for word in german.split(' '):
            german_stemmed_word = self.german_stemmer.stem(word.strip().decode('utf-8')).encode('utf-8')
            self.german_vocab.add(german_stemmed_word)
        for word in english.split(' '):
            english_stemmed_word = self.english_stemmer.stem(word.strip().decode('utf-8')).encode('utf-8')
            self.english_vocab.add(english_stemmed_word)
            for word in german.split(' '):
                german_stemmed_word = self.german_stemmer.stem(word.strip().decode('utf-8')).encode('utf-8')
                key = (german_stemmed_word, english_stemmed_word)
                if not self.translation_probs.has_key(key):
                    self.totals[english_stemmed_word] = self.totals.get(english_stemmed_word, 0) + 1.0
                self.translation_probs[key] = 1.0

    def normalize(self):
        for english_stemmed_word in self.english_vocab:
            for german_stemmed_word in self.german_vocab:
                key = (german_stemmed_word, english_stemmed_word)
                if self.translation_probs.has_key(key): #prevent populating entries unless they occur in parallel sentences
                    self.translation_probs[key] = self.translation_probs[key] / self.totals[english_stemmed_word] #totals of english word should NEVER be 0
                elif english_stemmed_word == self.null_val:
                    self.translation_probs[key] = 1.0 / len(self.german_vocab)

    def stop_condition(self, iter_count): # Currently only checking for iteration limit. Ideally, we should also check for
        # convergence, i.e., when parameters change by value below a certain threshold
        if iter_count == self.MAX_ITERS:
            return(True)
        else:
            return(False)


    def estimate_params(self):

        iter_count = 0

        """
        EM algorithm for estimating the translation probablities
        See https://www.cl.cam.ac.uk/teaching/1011/L102/clark-lecture3.pdf for a good tutorial
        """

        while(True):#until convergence or max_iters
            print("Iteration " + str(iter_count + 1))
            iter_count += 1
            self.counts = {} # All counts default to 0. These are counts of (german, english) word pairs
            self.totals = {} # All totals default to 0. These are sums of counts (marginalized over all foreign words), for each
            # english word
            with open(self.INPUT_FILE) as ip_file: 
                # Read one line at a time from file. No need to store file in memory
                for line in ip_file: 
                    # parse the line
                    german_sent,english_sent = self.get_parallel_instance(line) 

                    german_sent_dict = self.get_counts(german_sent)
                    english_sent_dict = self.get_counts(english_sent)
                    for german_word in german_sent_dict.keys(): # For each unique german word in the german sentence
                        german_word_count = german_sent_dict[german_word] # Its count in the german sentence
                        total_s = 0.0  # Expected count of number of alignments for this german word with any english word
                        for english_word in english_sent_dict.keys():
                            total_s += self.translation_probs.get((german_word, english_word), 0.0) * german_word_count
                        for english_word in english_sent_dict.keys():
                            english_word_count  = english_sent_dict[english_word]
                            self.counts[(german_word, english_word)] = self.counts.get((german_word, english_word), 0.0) + self.translation_probs.get((german_word, english_word), 0.0) * german_word_count * english_word_count / total_s
                            # Expected count of alignments between german word and this english word, divided by the expected count of all alignments of this german word
                            self.totals[english_word] = self.totals.get(english_word, 0.0) + self.translation_probs.get((german_word, english_word), 0.0)* german_word_count * english_word_count / total_s
                            # Aggregating the expected counts of all german words, for each english word. This will be used as a normalizing factor
                for english_word in self.totals.keys(): # restricting to domain total( . )
                    for word_pair in self.counts.keys():
                        german_word = word_pair[0]
                        if(word_pair[1] != english_word): # restricting to domain count( . | e )
                            continue
                        self.translation_probs[word_pair] = self.counts.get((german_word, english_word), 0.0) / self.totals.get(english_word, 0.0)
                        # Neither domain nor counts should never be 0 given our domain restriction

            if(iter_count % 2 == 0): #Store the model every other iteration
                print("Storing model after %d iterations"%(iter_count))
                model_dump = open(self.OUTPUT_FILE, 'wb')
                pickle.dump(self.translation_probs, model_dump)
                model_dump.close()

            if(self.stop_condition(iter_count)):
                print("Storing model after %d iterations" % (iter_count))
                model_dump = open(self.OUTPUT_FILE, 'wb')
                pickle.dump(self.translation_probs, model_dump)
                model_dump.close()
                break

        print("Memory usage stats")
        print("German vocab length: ", len(self.german_vocab))
        print("English vocab length: ", len(self.english_vocab))
        print("No of cross product entries required: ", len(self.german_vocab) * len(self.english_vocab))

        print("Num of conditional probabilities actually stored: ", len(self.translation_probs.keys()))
        print("Num of counts actually stored: ", len(self.counts.keys()))
        print("Num of totals actually stored: ", len(self.totals.keys()))

        self.sanity_check()


    def sanity_check(self, n_sample = None):
        if n_sample is None:
            test_english_words = [english_word for english_word in self.english_vocab] #should not further stem words in english vocab
            print("Performing sanity check on full vocabulary")
        else:
            test_english_words = [
                english_word for
                english_word in random.sample(self.english_vocab, n_sample)]
            print("Spot checking for the following English words ")
            print(test_english_words)

        for english_word in test_english_words:
            max_prob_word = None
            max_prob = 0.0
            tot_conditional_prob = 0.0
            for german_word in self.german_vocab:
                if self.translation_probs.get((german_word,english_word), 0.0) != 0.0:
                    tot_conditional_prob += self.translation_probs.get((german_word, english_word), 0.0)
                    if self.translation_probs.get((german_word,english_word), 0.0) > max_prob:
                        max_prob = self.translation_probs.get((german_word,english_word), 0.0)
                        max_prob_word = german_word
            assert abs(tot_conditional_prob - 1.0) < 0.000000000001, 'Tot conditional probability != 1 !!!'
            if n_sample is not None:
                print("Most likely word for ", english_word, " is the german word ", max_prob_word, " with translation probability ", max_prob)
        print("Sanity check passed!")

