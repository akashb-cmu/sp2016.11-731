def get_hypotheses(hyp_file_path, gold_file_path):
    corpus = []
    labels = []
    with open(hyp_file_path, 'r') as hyp_file:
        for line in hyp_file:
            if len(line.strip(' \t\r\n')) > 0:
                instance = (sub_str.strip(' \t\r\n') for sub_str in line.split('|||'))
                corpus.append(instance)
    with open(gold_file_path, 'r') as gold_file:
        for line in gold_file:
            if len(line.strip(' \t\r\n')) > 0:
                label = int(line.strip(' \t\r\n'))
                labels.append(label)
    assert(len(labels) == len(corpus[:len(labels)])), "Len of corpus and labels doesn't match"
    return([zip(corpus[:len(labels)], labels), corpus[len(labels):]])
