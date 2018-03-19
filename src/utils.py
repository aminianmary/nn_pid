from collections import Counter, defaultdict
import re, codecs, random
import numpy as np

class ConllStruct:
    def __init__(self, entries):
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def get_predicates(self):
        predicates =[]
        for e in self.entries:
            if e.is_pred:
                predicates.append(e.id)
        return predicates



class ConllEntry:
    def __init__(self, id, form, lemma, pos, sense='_', parent_id='_', relation='_', predicateList=dict(),
                 is_pred=False):
        self.id = id
        self.form = form
        self.lemma = lemma
        self.norm = normalize(form)
        self.lemmaNorm = normalize(lemma)
        self.pos = pos.upper()
        self.parent_id = parent_id
        self.relation = relation
        self.predicateList = predicateList
        self.sense = sense
        self.is_pred = is_pred

    def __str__(self):
        entry_list = [str(self.id+1), self.form, self.lemma, self.lemma, self.pos, self.pos, '_', '_',
                      self.parent_id,
                      self.parent_id, self.relation, self.relation,
                      'Y' if self.is_pred else '_',
                      self.sense if self.is_pred else '_']
        return '\t'.join(entry_list)

def vocab(sentences, min_count=2):
    wordsCount = Counter()
    posCount = Counter()
    semRelCount = Counter()
    lemma_count = Counter()
    chars = set()

    for sentence in sentences:
        wordsCount.update([node.norm for node in sentence.entries])
        posCount.update([node.pos for node in sentence.entries])
        for node in sentence.entries:
            if node.predicateList == None:
                continue
            if node.is_pred:
                lemma_count.update([node.lemma])
            for pred in node.predicateList.values():
                if pred!='?':
                    semRelCount.update([pred])
            for c in list(node.norm):
                chars.add(c.lower())

    words = set()
    for w in wordsCount.keys():
        if wordsCount[w] >= min_count:
            words.add(w)
    lemmas = set()
    for l in lemma_count.keys():
        if lemma_count[l] >= min_count:
            lemmas.add(l)
    return (list(words), list(lemmas),
            list(posCount), list(semRelCount.keys()), list(chars))

def read_conll(fh):
    sentences = codecs.open(fh, 'r').read().strip().split('\n\n')
    read = 0
    for sentence in sentences:
        words = []
        entries = sentence.strip().split('\n')
        for entry in entries:
            spl = entry.split('\t')
            predicateList = dict()
            is_pred = False
            if spl[12] == 'Y':
                is_pred = True

            for i in range(14, len(spl)):
                predicateList[i - 14] = spl[i]

            words.append(
                ConllEntry(int(spl[0]) - 1, spl[1], spl[3], spl[5], spl[13], spl[9], spl[11], predicateList,
                           is_pred))
        read += 1
        yield ConllStruct(words)
    print read, 'sentences read.'

def write_conll(fn, conll_structs):
    with codecs.open(fn, 'w') as fh:
        for conll_struct in conll_structs:
            predicates = conll_struct.get_predicates()
            args = ['_' for _ in xrange(len(predicates))]
            for i in xrange(len(conll_struct.entries)):
                entry = conll_struct.entries[i]
                fh.write(str(entry))
                if len(predicates)!=0:
                    fh.write('\t'+'\t'.join(args))
                fh.write('\n')
            fh.write('\n')

numberRegex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+")
urlRegex = re.compile("((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)")

def normalize(word):
    return '<NUM>' if numberRegex.match(word) else ('<URL>' if urlRegex.match(word) else word.lower())

def get_batches(buckets, model, is_train):
    d_copy = [buckets[i][:] for i in range(len(buckets))]
    if is_train:
        for dc in d_copy:
            random.shuffle(dc)
    mini_batches = []
    batch, cur_len, cur_c_len = [], 0, 0
    for dc in d_copy:
        for d in dc:
            if (is_train and len(d)<=100) or not is_train:
                batch.append(d.entries)
                cur_c_len = max(cur_c_len, max([len(w.norm) for w in d.entries]))
                cur_len = max(cur_len, len(d))

            if cur_len * len(batch) >= model.options.batch:
                add_to_minibatch(batch, cur_c_len, cur_len, mini_batches, model)
                batch, cur_len, cur_c_len = [], 0, 0

    if len(batch)>0 and not is_train:
        add_to_minibatch(batch, cur_c_len, cur_len, mini_batches, model)
    if is_train:
        random.shuffle(mini_batches)
    return mini_batches

def add_to_minibatch(batch, max_char_len, max_w_len, mini_batches, model):
    num_sen = len(batch)
    words = np.array([np.array(
        [model.word_dict.get(batch[i][j].norm, model.unk_id) if j < len(batch[i]) else model.PAD for i in range(num_sen)]) for j in range(max_w_len)])
    pwords = np.array([np.array(
        [model.x_pe_dict.get(batch[i][j].norm, model.unk_id) if j < len(batch[i]) else model.PAD for i in
         range(num_sen)]) for j in range(max_w_len)])
    pos = np.array([np.array(
        [model.pos_dict.get(batch[i][j].pos, model.unk_id) if j < len(batch[i]) else model.PAD for i in
         range(num_sen)]) for j in range(max_w_len)])

    # For all characters in all words in all sentences, put them all in a tensor except the ones that are outside the boundary of the sentence.
    # todo: fix this for other branches as well.
    chars = [list() for _ in range(max_char_len)]
    for c_position in range(max_char_len):
        ch = [model.PAD]*(num_sen * max_w_len)
        offset = 0
        for word_position in range(max_w_len):
            for sen_position in range(num_sen):
                if word_position<len(batch[sen_position]) and c_position<len(batch[sen_position][word_position].norm):
                    ch[offset] = model.char_dict.get(batch[sen_position][word_position].norm[c_position], model.unk_id)
                offset+=1
        chars[c_position] = np.array(ch)
    chars = np.array(chars)

    roles = np.array([np.array([1 if j < len(batch[i]) and batch[i][j].is_pred else 0 for i in range(len(batch))]) for j in range(max_w_len)])
    masks = np.array([np.array([1 if j < len(batch[i]) else 0 for i in range(num_sen)]) for j in range(max_w_len)])

    mini_batches.append((words, pwords, pos, chars, roles, masks))

def evaluate(output, gold):
    tp, fp, tn, fn = 0, 0, 0, 0
    gold_sentences = read_conll(gold)
    auto_sentences = read_conll(output)

    for g_s, a_s in zip (gold_sentences, auto_sentences):
        g_predicates = g_s.get_predicates()
        a_predicates = a_s.get_predicates()

        for a_p in a_predicates:
            if a_p in g_predicates:
                tp+=1
            else:
                fp+=1
        for g_p in g_predicates:
            if not g_p in a_predicates:
                fn+=1
    precision = float (tp)/(tp+fp) if tp+fp!=0 else 0.0
    recall = float (tp)/(tp+fn) if tp+fn!=0 else 0.0
    f_score = 2*(precision*recall)/(precision+recall) if precision+recall!=0 else 0
    return precision,recall,f_score

