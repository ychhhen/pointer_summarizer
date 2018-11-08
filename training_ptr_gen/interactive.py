from __future__ import division, print_function, unicode_literals

import os
import struct
import sys

import torch
from tensorflow.core.example import example_pb2
from torch.autograd import Variable

from data_util import config, data
from data_util.batcher import Batcher
from data_util.data import Vocab
from log_util import get_logger
from model import Model
from train_util import get_input_from_batch

# sys.setdefaultencoding('utf8')

use_cuda = config.use_gpu and torch.cuda.is_available()
LOGGER = get_logger('pointer.generator.interactive')


class Beam(object):
    def __init__(self, tokens, log_probs, state, context, coverage):
        self.tokens = tokens
        self.log_probs = log_probs
        self.state = state
        self.context = context
        self.coverage = coverage

    def extend(self, token, log_prob, state, context, coverage):
        return Beam(
            tokens=self.tokens + [token],
            log_probs=self.log_probs + [log_prob],
            state=state,
            context=context,
            coverage=coverage)

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def avg_log_prob(self):
        return sum(self.log_probs) / len(self.tokens)


class BeamSearchInteractive(object):
    def __init__(self, model_file_path):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)

        self.model = Model(model_file_path, is_eval=True)

    def create_batched_input(self, document, summary=''):
        tf_example = example_pb2.Example()
        tf_example.features.feature['article'].bytes_list.value.extend(
            [document.encode()])
        tf_example.features.feature['abstract'].bytes_list.value.extend(
            [summary.encode()])
        tf_example_str = tf_example.SerializeToString()
        str_len = len(tf_example_str)
        with open(os.path.join(config.log_root, 'batch.bin'), 'w') as fw:
            fw.write(struct.pack('q', str_len))
            fw.write(struct.pack('%ds' % str_len, tf_example_str))
        batcher = Batcher(
            os.path.join(config.log_root, 'batch.bin'),
            self.vocab,
            mode='decode',
            batch_size=config.beam_size,
            single_pass=True)
        return batcher

    def sort_beams(self, beams):
        return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)

    def predict(self, sentence):
        # Create a batcher object
        batcher = self.create_batched_input(sentence)
        # Obtain the batch
        batch = batcher.next_batch()
        # Run beam search to get best Hypothesis
        best_summary = self.beam_search(batch)

        # Extract the output ids from the hypothesis and
        # convert back to words
        output_ids = [int(t) for t in best_summary.tokens[1:]]
        decoded_words = data.outputids2words(
            output_ids, self.vocab,
            (batch.art_oovs[0] if config.pointer_gen else None))

        # Remove the [STOP] token from decoded_words, if necessary
        try:
            fst_stop_idx = decoded_words.index(data.STOP_DECODING)
            decoded_words = decoded_words[:fst_stop_idx]
        except ValueError:
            decoded_words = decoded_words

        LOGGER.debug('Decoded words = {}'.format(' '.join(decoded_words)))

        os.remove(os.path.join(config.log_root, 'batch.bin'))

        return decoded_words

    def beam_search(self, batch):
        # batch should have only one example
        (enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab,
         extra_zeros, c_t_0, coverage_t_0) = get_input_from_batch(
             batch, use_cuda)

        (encoder_outputs, encoder_feature,
         encoder_hidden) = self.model.encoder(enc_batch, enc_lens)
        s_t_0 = self.model.reduce_state(encoder_hidden)

        dec_h, dec_c = s_t_0  # 1 x 2*hidden_size
        dec_h = dec_h.squeeze()
        dec_c = dec_c.squeeze()

        # decoder batch preparation, it has beam_size example initially
        # everything is repeated
        beams = [
            Beam(
                tokens=[self.vocab.word2id(data.START_DECODING)],
                log_probs=[0.0],
                state=(dec_h[0], dec_c[0]),
                context=c_t_0[0],
                coverage=(coverage_t_0[0] if config.is_coverage else None))
            for _ in range(config.beam_size)
        ]
        results = []
        steps = 0
        while steps < config.max_dec_steps and len(results) < config.beam_size:
            latest_tokens = [h.latest_token for h in beams]
            latest_tokens = [
                t if t < self.vocab.size() else self.vocab.word2id(
                    data.UNKNOWN_TOKEN) for t in latest_tokens
            ]
            y_t_1 = Variable(torch.LongTensor(latest_tokens))
            if use_cuda:
                y_t_1 = y_t_1.cuda()
            all_state_h = []
            all_state_c = []

            all_context = []

            for h in beams:
                state_h, state_c = h.state
                all_state_h.append(state_h)
                all_state_c.append(state_c)

                all_context.append(h.context)

            s_t_1 = (torch.stack(all_state_h, 0).unsqueeze(0),
                     torch.stack(all_state_c, 0).unsqueeze(0))
            c_t_1 = torch.stack(all_context, 0)

            coverage_t_1 = None
            if config.is_coverage:
                all_coverage = []
                for h in beams:
                    all_coverage.append(h.coverage)
                coverage_t_1 = torch.stack(all_coverage, 0)

            (final_dist, s_t, c_t,
             attn_dist, p_gen, coverage_t) = self.model.decoder(
                 y_t_1, s_t_1, encoder_outputs, encoder_feature,
                 enc_padding_mask, c_t_1, extra_zeros, enc_batch_extend_vocab,
                 coverage_t_1, steps)

            topk_log_probs, topk_ids = torch.topk(final_dist,
                                                  config.beam_size * 2)

            dec_h, dec_c = s_t
            dec_h = dec_h.squeeze()
            dec_c = dec_c.squeeze()

            all_beams = []
            num_orig_beams = 1 if steps == 0 else len(beams)
            for i in range(num_orig_beams):
                h = beams[i]
                state_i = (dec_h[i], dec_c[i])
                context_i = c_t[i]
                coverage_i = (coverage_t[i] if config.is_coverage else None)

                # for each of the top 2*beam_size hyps:
                for j in range(config.beam_size * 2):
                    new_beam = h.extend(
                        token=topk_ids[i, j].item(),
                        log_prob=topk_log_probs[i, j].item(),
                        state=state_i,
                        context=context_i,
                        coverage=coverage_i)
                    all_beams.append(new_beam)

            beams = []
            for h in self.sort_beams(all_beams):
                if h.latest_token == self.vocab.word2id(data.STOP_DECODING):
                    if steps >= config.min_dec_steps:
                        results.append(h)
                else:
                    beams.append(h)
                if len(beams) == config.beam_size or len(
                        results) == config.beam_size:
                    break

            steps += 1

        if len(results) == 0:
            results = beams

        beams_sorted = self.sort_beams(results)

        return beams_sorted[0]


if __name__ == '__main__':
    model_filename = sys.argv[1]
    beam_search_processor = BeamSearchInteractive(model_filename)

    while True:
        print("Input sentence at the prompt: ")
        sentence = raw_input()
        print()
        if sentence.strip() == "exit":
            sys.exit(0)
        decoded_words = beam_search_processor.predict(sentence)
        print()
        print("Decoded words: %s" % ' '.join(decoded_words))
        print("==============================")
        print()

