import numpy as np

def get_data_and_w2vec_dicts(hyp_file_path, gold_file_path): #, w2vec_model):
    corpus = []
    labels = []
    #relevant_w2vec_model = []
    word2index = {}
    index2word = {}
    max_word_index = 1
    with open(hyp_file_path, 'r') as hyp_file:
        for line in hyp_file:
            if len(line.strip(' \t\r\n')) > 0:
                instance = []
                for sentence in line.split('|||'):
                    sentence = sentence.strip(" \t\r\n").lower()
                    instance.append(sentence)
                    for sentence_index, word in enumerate(sentence.split(" ")):
                        if word2index.get(word, None) is None:
                            word2index[word] = max_word_index
                            index2word[max_word_index] = word
                            """
                            try:
                                #relevant_w2vec_model[word] = w2vec_model[word]
                                relevant_w2vec_model.append(w2vec_model[word])
                            except:
                                #relevant_w2vec_model[word] = np.zeros(get_wvec_size(w2vec_model))
                                relevant_w2vec_model.append(np.zeros(get_wvec_size(w2vec_model)))
                            assert(len(relevant_w2vec_model) == max_word_index), "Index and embedding matrix length out of sync"
                            """
                            max_word_index += 1
                corpus.append(instance)
    with open(gold_file_path, 'r') as gold_file:
        for line in gold_file:
            if len(line.strip(' \t\r\n')) > 0:
                label = int(line.strip(' \t\r\n'))
                labels.append(label)
    assert(len(labels) == len(corpus[:len(labels)])), "Len of corpus and labels doesn't match"
    return ([zip(corpus[:len(labels)], labels), corpus[len(labels):], word2index, index2word])
    #return([zip(corpus[:len(labels)], labels), corpus[len(labels):], np.array(relevant_w2vec_model, dtype=np.float64), word2index, index2word])

def get_wvec_size(w2vec_model):
    return(len(w2vec_model['test']))

def get_sentence_vector(sentence, max_len, word2index):
    word_list = sentence.split(" ")
    ret_vect = np.zeros(max_len, dtype=np.int32)
    for w_index  in range(max_len):
        if w_index < len(word_list):
            word = word_list[w_index]
            word_index = word2index.get(word, 0)
            assert word_index != 0, "Word not found!"
        else:
            word_index = 0
        ret_vect[w_index] = word_index
    assert len(ret_vect) == max_len, "Returning mat of len < max_len"
    return(ret_vect)

def get_matrized_data(labeled_instances, word2index, train_split_ratio):
    max_ref_len = max(len(t_instance[0][2].split(" ")) for t_instance in labeled_instances)
    max_hyp1_len = max(len(t_instance[0][0].split(" ")) for t_instance in labeled_instances)
    max_hyp2_len = max(len(t_instance[0][1].split(" ")) for t_instance in labeled_instances)

    max_ref_len = max_hyp1_len = max_hyp2_len = max(max_ref_len, max_hyp1_len, max_hyp2_len)
    print("Initializing tensors assuming index representation of word vectors")

    ref_tensor = np.zeros((2*len(labeled_instances), max_ref_len), dtype=np.int32)
    hyp1_tensor = np.zeros((2*len(labeled_instances), max_hyp1_len), dtype=np.int32)
    hyp2_tensor = np.zeros((2*len(labeled_instances), max_hyp2_len), dtype=np.int32)

    print("Initializing labels")

    #hyp1_2way_labels = np.zeros(len(labeled_instances), dtype=np.int32)
    #hyp2_2way_labels = np.zeros(len(labeled_instances), dtype=np.int32)
    common_3way_labels = np.zeros((2*len(labeled_instances), 3), dtype=np.int32)

    print("Building tensors")

    for instance_index, instance in enumerate(labeled_instances):
        [hyp1, hyp2, ref] = instance[0]
        label = instance[1]
        hyp1_sentence_vector = get_sentence_vector(hyp1, max_hyp1_len, word2index)
        hyp2_sentence_vector = get_sentence_vector(hyp2, max_hyp2_len, word2index)
        ref_sentence_vector = get_sentence_vector(ref, max_ref_len, word2index)

        hyp1_tensor[2*instance_index] = hyp1_sentence_vector
        hyp1_tensor[2 * instance_index + 1] = hyp2_sentence_vector
        hyp2_tensor[instance_index] = hyp2_sentence_vector
        hyp2_tensor[2 * instance_index + 1] = hyp1_sentence_vector
        ref_tensor[2*instance_index] = ref_sentence_vector
        ref_tensor[2*instance_index + 1] = ref_sentence_vector

        # common_3way_labels[instance_index] = label

        if label == -1:
            common_3way_labels[2*instance_index] = [1,0,0]
            common_3way_labels[2 * instance_index + 1] = [0, 0, 1]
        elif label == 0:
            common_3way_labels[2*instance_index] = [0,1,0]
            common_3way_labels[2 * instance_index + 1] = [0, 1, 0]
        else:
            common_3way_labels[2*instance_index] = [0,0,1]
            common_3way_labels[2*instance_index + 1] = [1, 0, 0]

        """
        if label == -1:
            hyp1_2way_labels[instance_index] = 1
            hyp1_2way_labels[instance_index] = 0
        elif label == 1:
            hyp1_2way_labels[instance_index] = 0
            hyp2_2way_labels[instance_index] = 1
        else:
            hyp2_2way_labels[instance_index] = hyp1_2way_labels[instance_index] = 1
        """
    train_split_index = 2*int(len(labeled_instances) * train_split_ratio)
    print("Building train split")
    train_split = [elt[:train_split_index] for elt in
                   [ref_tensor, hyp1_tensor, hyp2_tensor, common_3way_labels]]
    print("Building test split")
    test_split = [elt[train_split_index:] for elt in
                   [ref_tensor, hyp1_tensor, hyp2_tensor, common_3way_labels]]
    return([train_split, test_split, max_ref_len])

