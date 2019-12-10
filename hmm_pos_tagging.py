import math
from collections import defaultdict

def load_corpus(path):
    corpus = {}
    i = 0
    with open(path, 'r') as file:
        for line in file.readlines():
            corpus[i] = [tuple(element.split('=')) for element in line.split()]
            i = i+1
    return corpus


class Tagger(object):

    def __init__(self, sentences):
        self.init_probs = {}
        self.tag = defaultdict(int)
        self.trans_probs = defaultdict(lambda: defaultdict(float))
        self.em_probs = defaultdict(lambda: defaultdict(float))
        self.token = set()
        self.smoothing = 1e-10
        prev_tag = 'prev_tag'
        for token in sentences.values():
            if token[0][1] not in self.init_probs:
                self.init_probs[token[0][1]] = 1
            else:
                self.init_probs[token[0][1]] += 1
            for element in token:
                self.tag[element[1]] += 1
                self.em_probs[element[1]][element[0]] += 1
                self.token.add(element[0])
                if prev_tag == 'prev_tag':
                    pass
                else:
                    self.trans_probs[prev_tag][element[1]] += 1
                prev_tag = element[1]
        for tag, frequency in self.init_probs.items():
            self.init_probs[tag] = math.log((frequency + self.smoothing) /
                                            (len(sentences) + self.smoothing * len(self.tag)))
        for prev, post_dict in self.trans_probs.items():
            sum_frequency, prev_domain = sum(post_dict.values()), len(post_dict)
            for post_tag, frequency in post_dict.items():
                self.trans_probs[prev][post_tag] = math.log((frequency + self.smoothing) /
                                                            (sum_frequency + self.smoothing * prev_domain))
        copy_em_prob = self.em_probs.copy()
        for tag, word_dict in copy_em_prob.items():
            domain = len(word_dict)
            count_tag = sum(word_dict.values())
            for word, frequency in word_dict.items():
                self.em_probs[tag][word] = math.log((frequency+self.smoothing)
                                                    / (count_tag+self.smoothing*(domain + 1)))
            self.em_probs[tag]['<UNK>'] = math.log(self.smoothing/(count_tag + self.smoothing * (domain + 1)))

    def most_probable_tags(self, tokens):
        tag_with_max_prob = ''
        result_list = []
        for token in tokens:
            max_prob = -999999
            for tag, word_dict in self.em_probs.items():
                if token in word_dict and max_prob < self.em_probs[tag][token]:
                    tag_with_max_prob = tag
                    max_prob = self.em_probs[tag][token]
                elif token not in word_dict and max_prob < self.em_probs[tag]['<UNK>']:
                    tag_with_max_prob = 'X'
                    max_prob = self.em_probs[tag]['<UNK>']
            result_list.append(tag_with_max_prob)
        return result_list

#apply viterbi algorithm
    def viterbi_tags(self, tokens):
        delta = [{} for i in range(len(tokens))]
        path = []
        max_prob_tag = ' '
        for tag, prob in self.init_probs.items():
            if tokens[0] in self.em_probs[tag]:
                delta[0][tag] = prob + self.em_probs[tag][tokens[0]]
            else:
                delta[0][tag] = prob + self.em_probs[tag]['<UNK>']
        for i in range(1, len(tokens)):
            for current_tag in self.tag:
                max_prob = -99999
                for prev_tag, prev_prob in delta[i-1].items():
                    if max_prob < (self.trans_probs[prev_tag][current_tag] + prev_prob):
                        max_prob = self.trans_probs[prev_tag][current_tag] + prev_prob
                if tokens[i] in self.em_probs[current_tag]:
                    delta[i][current_tag] = max_prob + self.em_probs[current_tag][tokens[i]]
                else:
                    delta[i][current_tag] = max_prob + self.em_probs[current_tag]['<UNK>']
        for tag_prob_dict in delta:
            max_prob = -99999
            for tag, prob in tag_prob_dict.items():
                if prob > max_prob:
                    max_prob = prob
                    max_prob_tag = tag
            path.append(max_prob_tag)
        return path
