from dynet import *
from utils import read_conll, get_batches, evaluate, write_conll
import time, random, os,math
import numpy as np


class SRLLSTM:
    def __init__(self, words, lemmas, pos, roles, chars, options):
        self.model = Model()
        self.options = options
        self.batch_size = options.batch
        self.trainer = AdamTrainer(self.model, options.learning_rate, options.beta1, options.beta2, options.eps)
        self.trainer.set_clip_threshold(1.0)
        self.unk_id = 0
        self.PAD = 1
        self.NO_LEMMA = 2
        self.word_dict = {word: ind + 2 for ind, word in enumerate(words)}
        self.pos_dict = {p: ind + 2 for ind, p in enumerate(pos)}
        self.ipos = ['<UNK>', '<PAD>'] + pos
        self.roles = {r: ind for ind, r in enumerate(roles)} #todo
        self.iroles = roles #todo
        self.char_dict = {c: i + 2 for i, c in enumerate(chars)}
        self.d_w = options.d_w
        self.d_h = options.d_h
        self.d_cw = options.d_cw
        self.d_pos = options.d_pos
        self.k = options.k
        self.alpha = options.alpha
        self.external_embedding = None
        self.x_pe = None
        external_embedding_fp = open(options.external_embedding, 'r')
        external_embedding_fp.readline()
        self.external_embedding = {line.split(' ')[0]: [float(f) for f in line.strip().split(' ')[1:]] for line in external_embedding_fp}
        external_embedding_fp.close()
        self.edim = len(self.external_embedding.values()[0])
        self.noextrn = [0.0 for _ in xrange(self.edim)]
        self.x_pe_dict = {word: i + 2 for i, word in enumerate(self.external_embedding)}
        self.x_pe = self.model.add_lookup_parameters((len(self.external_embedding) + 2, self.edim))
        for word, i in self.x_pe_dict.iteritems():
            self.x_pe.init_row(i, self.external_embedding[word])
        self.x_pe.init_row(0,self.noextrn)
        self.x_pe.init_row(1,self.noextrn)
        self.x_pe.set_updated(False)

        print 'Load external embedding. Vector dimensions', self.edim

        self.inp_dim = self.d_w + self.d_cw + self.d_pos + (self.edim if self.external_embedding is not None else 0)
        self.char_lstm = BiRNNBuilder(1, options.d_c, options.d_cw, self.model, VanillaLSTMBuilder)
        self.deep_lstms = BiRNNBuilder(self.k, self.inp_dim, 2*self.d_h, self.model, VanillaLSTMBuilder)
        self.out_layer = self.model.add_parameters((2, 2*self.d_h))
        self.out_bias = self.model.add_parameters((2, ), init = ConstInitializer(0))
        self.x_re = self.model.add_lookup_parameters((len(self.word_dict) + 2, self.d_w))
        self.ce = self.model.add_lookup_parameters((len(chars) + 2, options.d_c)) # character embedding
        self.x_pos = self.model.add_lookup_parameters((len(pos)+2, self.d_pos))

    def Save(self, filename):
        self.model.save(filename)

    def Load(self, filename):
        self.model.populate(filename)

    def rnn(self, words, pwords, pos, chars):
        cembed = [lookup_batch(self.ce, c) for c in chars]
        char_fwd, char_bckd = self.char_lstm.builder_layers[0][0].initial_state().transduce(cembed)[-1], \
                              self.char_lstm.builder_layers[0][1].initial_state().transduce(reversed(cembed))[-1]
        crnn = reshape(concatenate_cols([char_fwd, char_bckd]), (self.d_cw, words.shape[0] * words.shape[1]))
        cnn_reps = [list() for _ in range(len(words))]

        for i in range(words.shape[0]):
            cnn_reps[i] = pick_batch(crnn, [i * words.shape[1] + j for j in range(words.shape[1])], 1)

        inputs = [concatenate([lookup_batch(self.x_re, words[i]), lookup_batch(self.x_pe, pwords[i]), lookup_batch(self.x_pos, pos[i]), cnn_reps[i]]) for i in range(len(words))]

        for fb, bb in self.deep_lstms.builder_layers:
            f, b = fb.initial_state(), bb.initial_state()
            fs, bs = f.transduce(inputs), b.transduce(reversed(inputs))
            inputs = [concatenate([f, b]) for f, b in zip(fs, reversed(bs))]
        return inputs

    def buildGraph(self, minibatch):
        words, pwords, pos, chars, roles, masks = minibatch
        bilstms = self.rnn(words, pwords, pos, chars)
        bilstms_ = concatenate_cols(bilstms)
        output = affine_transform([self.out_bias.expr(), self.out_layer.expr(), bilstms_])
        dim_0_1 = output.dim()[0][1] if words.shape[0]!=1 else 1
        output_reshape = reshape(output, (output.dim()[0][0],), dim_0_1 * output.dim()[1])
        return output_reshape

    def decode(self, minibatches):
        outputs = []
        for b, batch in enumerate(minibatches):
            output = self.buildGraph(batch).npvalue().T
            argmax_vals = [np.argmax(o) for o in output]
            mask = batch[-1]
            num_sens, num_words = mask.shape[1], mask.shape[0]
            batch_output = [list() for _ in range(num_sens)]
            offset = 0
            for sen_index in range(num_sens):
                for word_index in range(num_words):
                    if mask[word_index][sen_index] != 0:
                        batch_output[sen_index].append(argmax_vals[offset])
                    offset += 1
            outputs += batch_output
            renew_cg()
        return outputs

    def Train(self, mini_batches, epoch, best_acc, options):
        print 'Start time', time.ctime()
        start = time.time()
        errs,loss,iters,sen_num = [],0,0,0
        dev_path = options.conll_dev

        part_size = max(1, len(mini_batches)/5)
        part = 0
        best_part = 0

        for b, mini_batch in enumerate(mini_batches):
            output = self.buildGraph(mini_batch)
            words, pwords, pos, chars, roles, masks = mini_batch
            num_roles = roles.shape[0] * roles.shape[1]
            roles_vec = np.reshape(roles.T, num_roles)
            masksTensor = reshape(inputTensor(np.reshape(masks.T, num_roles)), (1,), num_roles)
            sm_loss = pickneglogsoftmax_batch(output, roles_vec)
            masked_loss = cmult(sm_loss, masksTensor)
            loss_value = sum_batches(masked_loss)/num_roles
            loss_value.forward()
            loss += loss_value.value()
            loss_value.backward()
            self.trainer.update()
            renew_cg()
            print 'loss:', loss/(b+1), 'time:', time.time() - start, 'progress',round(100*float(b+1)/len(mini_batches),2),'%'
            loss, start = 0, time.time()
            errs, sen_num = [], 0
            iters+=1

            if (b+1)%part_size==0:
                part+=1

                if dev_path != '':
                    start = time.time()
                    write_conll(os.path.join(options.outdir, options.model) + str(epoch + 1) + "_" + str(part)+ '.txt',
                                self.Predict(dev_path))
                    precision,recall,fscore = evaluate(os.path.join(options.outdir, options.model) + str(epoch + 1) + "_" + str(part)+ '.txt',dev_path)
                    print 'Finished predicting dev on part '+ str(part)+ '; time:', time.time() - start
                    print 'epoch: ' + str(epoch) + ' part: '+ str(part) + '--precision: ' + str(precision)+' recall: '+str(recall)+' fscore: '+str(fscore)

                    if float(fscore) > best_acc:
                        self.Save(os.path.join(options.outdir, options.model))
                        best_acc = float(fscore)
                        best_part = part

        print 'best part on this epoch: '+ str(best_part)
        return best_acc

    def Predict(self, conll_path):
        print 'starting to decode...'
        dev_buckets = [list()]
        dev_data = list(read_conll(conll_path))
        for d in dev_data:
            dev_buckets[0].append(d)
        minibatches = get_batches(dev_buckets, self, False)
        results = self.decode(minibatches)
        for iSentence, sentence in enumerate(dev_data):
            for arg_index in xrange(len(sentence.entries)):
                prediction = True if results[iSentence][arg_index] == 1 else False
                sentence.entries[arg_index].is_pred = prediction
            yield sentence
