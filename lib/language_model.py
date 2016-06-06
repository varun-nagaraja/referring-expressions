# This code has been adapted from a version written by Jeff Donahue (jdonahue@eecs.berkeley.edu)

import math
import numpy as np
import random
import sys

sys.path.append('./caffe/python')
import caffe
from shared_utils import EOS_IDENTIFIER


class LanguageModel:
  def __init__(self, weights_path, lstm_net_proto, vocab_path, device_id=-1):
    if device_id >= 0:
      caffe.set_mode_gpu()
      caffe.set_device(device_id)
    else:
      caffe.set_mode_cpu()
    phase = caffe.TEST
    # Setup sentence prediction net.
    self.lstm_net = caffe.Net(lstm_net_proto, weights_path, phase)
    self.vocab_inverted = []
    self.vocab = {}
    with open(vocab_path, 'r') as vocab_file:
      self.vocab_inverted += [word.strip() for word in vocab_file.readlines()]
    for (index, val) in enumerate(self.vocab_inverted):
      self.vocab[val] = index

    net_vocab_size = self.lstm_net.blobs['predict'].data.shape[2]
    if len(self.vocab_inverted) != net_vocab_size:
      raise Exception('Invalid vocab file: contains %d words; '
        'net expects vocab with %d words' % (len(self.vocab_inverted), net_vocab_size))

  def sentence(self, vocab_indices):
    """
    Convert word embedding to human readable sentences
    """
    sentence = ' '.join([self.vocab_inverted[i] for i in vocab_indices])
    if not sentence: return sentence
    sentence = sentence[0].upper() + sentence[1:]
    # If sentence ends with ' <EOS>', remove and replace with '.'
    # Otherwise (doesn't end with '<EOS>' -- maybe was the max length?):
    # append '...'
    suffix = ' ' + EOS_IDENTIFIER
    if sentence.endswith(suffix):
      sentence = sentence[:-len(suffix)] + '.'
    else:
      sentence += '...'
    return sentence

  def caption_batch_size(self):
    return self.lstm_net.blobs['timestep_cont'].data.shape[1]

  def set_caption_batch_size(self, batch_size):
    self.lstm_net.blobs['timestep_cont'].reshape(1, batch_size)
    if len(self.lstm_net.blobs['timestep_input'].shape) == 3:
      self.lstm_net.blobs['timestep_input'].reshape(1, batch_size, self.lstm_net.blobs['timestep_input'].shape[2])
    else:
      self.lstm_net.blobs['timestep_input'].reshape(1, batch_size)
    self.lstm_net.blobs['fc7_img'].reshape(batch_size, *self.lstm_net.blobs['fc7_img'].data.shape[1:])
    self.lstm_net.blobs['fc7_obj'].reshape(batch_size, *self.lstm_net.blobs['fc7_obj'].data.shape[1:])
    self.lstm_net.blobs['bbox_features'].reshape(batch_size, *self.lstm_net.blobs['bbox_features'].data.shape[1:])
    self.lstm_net.reshape()


  # Sample with temperature temp.
  # Or score caption given as prefix words.
  def sample_captions(self, fc7_obj, fc7_img, bbox_features, prefix_words=[],
                      prob_output_name='probs',pred_output_name='predict', temp=float("inf"), max_length=20):
    net = self.lstm_net
    batch_size = fc7_img.shape[0]
    self.set_caption_batch_size(batch_size)
    cont_input = np.zeros_like(net.blobs['timestep_cont'].data)
    word_input = np.zeros(net.blobs['timestep_input'].data.shape[0:2])
    fc7_img = np.asarray(fc7_img)
    fc7_obj = np.asarray(fc7_obj)
    bbox_features = np.asarray(bbox_features)
    output_captions = [[] for _ in range(batch_size)]
    output_probs = [[] for _ in range(batch_size)]
    caption_index = 0
    num_done = 0
    while num_done < batch_size and caption_index < max_length:
      if caption_index == 0:
        cont_input[:] = 0
      elif caption_index == 1:
        cont_input[:] = 1
      if caption_index == 0:
        word_input[:] = self.vocab[EOS_IDENTIFIER]
      else:
        for index in range(batch_size):
          word_input[0, index] = \
              output_captions[index][caption_index - 1] if \
              caption_index <= len(output_captions[index]) else self.vocab[EOS_IDENTIFIER]

      net.forward(timestep_input=word_input, timestep_cont=cont_input,
                  fc7_img=fc7_img, fc7_obj=fc7_obj, bbox_features=bbox_features)
      net_output_probs = net.blobs[prob_output_name].data[0]
      net_output_preds = net.blobs[pred_output_name].data[0]
      if temp == 1.0 or temp == float('inf'):
        samples = [
            random_choice_from_probs(dist, temp=temp, already_softmaxed=True)
            for dist in net_output_probs
        ]
      else:
        samples = [
            random_choice_from_probs(preds, temp=temp, already_softmaxed=False)
            for preds in net_output_preds
        ]

      for index, next_word_sample in enumerate(samples):
        # If the caption is empty, or non-empty but the last word isn't EOS,
        # predict another word.
        if not output_captions[index] or output_captions[index][-1] != self.vocab[EOS_IDENTIFIER]:
          if len(prefix_words) > 0 and caption_index < len(prefix_words[index]):
            # Copy word from prefix sentence
            output_captions[index].append(prefix_words[index][caption_index])
          else:
            output_captions[index].append(next_word_sample)
          word_prob = net_output_probs[index, output_captions[index][caption_index]]
          output_probs[index].append(word_prob)
          if output_captions[index][-1] == self.vocab[EOS_IDENTIFIER]: num_done += 1
      caption_index += 1
    return output_captions, output_probs


class MILContextLanguageModel(LanguageModel):
  def __init__(self, weights_path, lstm_net_proto, vocab_path, device_id=-1):
    LanguageModel.__init__(self, weights_path, lstm_net_proto, vocab_path, device_id=device_id)

  def set_caption_batch_size(self, batch_size):
    if len(self.lstm_net.blobs['timestep_input'].shape) == 3:
      self.lstm_net.blobs['timestep_input'].reshape(1, batch_size, self.lstm_net.blobs['timestep_input'].shape[2])
      self.lstm_net.blobs['timestep_cont'].reshape(1, batch_size, self.lstm_net.blobs['timestep_cont'].shape[2])
    else:
      self.lstm_net.blobs['timestep_input'].reshape(1, batch_size)
      self.lstm_net.blobs['timestep_cont'].reshape(1, batch_size)
    if 'fc7_obj' in self.lstm_net.blobs:
      self.lstm_net.blobs['fc7_obj'].reshape(batch_size, *self.lstm_net.blobs['fc7_obj'].data.shape[1:])
    if 'bbox_features' in self.lstm_net.blobs:
      self.lstm_net.blobs['bbox_features'].reshape(batch_size, *self.lstm_net.blobs['bbox_features'].data.shape[1:])
    if 'context_fc7' in self.lstm_net.blobs:
      self.lstm_net.blobs['context_fc7'].reshape(batch_size, *self.lstm_net.blobs['context_fc7'].data.shape[1:])
    if 'context_bbox_features' in self.lstm_net.blobs:
      self.lstm_net.blobs['context_bbox_features'].reshape(batch_size, *self.lstm_net.blobs['context_bbox_features'].data.shape[1:])

    self.lstm_net.reshape()

  # Sample with temperature temp.
  # Or score caption given as prefix words.
  def sample_captions_with_context(self, fc7_obj, bbox_features, context_fc7, context_bbox_features, prefix_words=[],
                      prob_output_name='probs',pred_output_name='r_mil_predict', temp=float("inf"), max_length=20):
    net = self.lstm_net
    batch_size = fc7_obj.shape[0]
    context_size = fc7_obj.shape[1]
    self.set_caption_batch_size(batch_size)
    cont_input = np.zeros_like(net.blobs['timestep_cont'].data)
    word_input = np.zeros_like(net.blobs['timestep_input'].data)
    fc7_obj = np.asarray(fc7_obj)
    bbox_features = np.asarray(bbox_features)
    output_captions = [[] for _ in range(batch_size)]
    output_probs = [[] for _ in range(batch_size)]
    output_all_probs = [[] for _ in range(batch_size*context_size)]
    context_inds = [[] for _ in range(batch_size)]
    caption_index = 0
    num_done = 0
    while num_done < batch_size and caption_index < max_length:
      if caption_index == 0:
        cont_input[:] = 0
      elif caption_index == 1:
        cont_input[:] = 1
      if caption_index == 0:
        word_input[:] = self.vocab[EOS_IDENTIFIER]
      else:
        for index in range(batch_size):
          word_input[0, index, :] = \
              output_captions[index][caption_index - 1] if \
              caption_index <= len(output_captions[index]) else self.vocab[EOS_IDENTIFIER]

      net.forward(timestep_input=word_input, timestep_cont=cont_input,fc7_obj=fc7_obj,bbox_features=bbox_features,
                  context_fc7=context_fc7, context_bbox_features=context_bbox_features)

      net_output_probs = net.blobs[prob_output_name].data[0]
      assert(net_output_probs.shape[0] == fc7_obj.shape[0])
      all_probs = net.blobs['all_probs'].data[0]
      assert(all_probs.shape[0] == batch_size*context_size)
      net_output_preds = net.blobs[pred_output_name].data[0]
      assert(net_output_preds.shape[0] == fc7_obj.shape[0])
      if temp == 1.0 or temp == float('inf'):
        samples = [
            random_choice_from_probs(dist, temp=temp, already_softmaxed=True)
            for dist in net_output_probs
        ]
      else:
        samples = [
            random_choice_from_probs(preds, temp=temp, already_softmaxed=False)
            for preds in net_output_preds
        ]
      context_preds = net.blobs['r_predict'].data[0][0]
      for index, next_word_sample in enumerate(samples):
        # If the caption is empty, or non-empty but the last word isn't EOS,
        # predict another word.
        if not output_captions[index] or output_captions[index][-1] != self.vocab[EOS_IDENTIFIER]:
          if len(prefix_words) > 0 and caption_index < len(prefix_words[index]):
            # Copy word from prefix sentence
            output_captions[index].append(prefix_words[index][caption_index])
          else:
            output_captions[index].append(next_word_sample)
          word_ind = output_captions[index][caption_index]
          word_prob = net_output_probs[index, word_ind]
          output_probs[index].append(word_prob)
          if output_captions[index][-1] == self.vocab[EOS_IDENTIFIER]: num_done += 1

          for other_box_ind in range(context_size):
            output_all_probs[index*context_size+other_box_ind].append(all_probs[index*context_size+other_box_ind,word_ind])

      caption_index += 1
    return output_captions, output_probs, output_all_probs


def softmax(softmax_inputs, temp):
  shifted_inputs = softmax_inputs - softmax_inputs.max()
  exp_outputs = np.exp(temp * shifted_inputs)
  exp_outputs_sum = exp_outputs.sum()
  if math.isnan(exp_outputs_sum):
    return exp_outputs * float('nan')
  assert exp_outputs_sum > 0
  if math.isinf(exp_outputs_sum):
    return np.zeros_like(exp_outputs)
  eps_sum = 1e-20
  return exp_outputs / max(exp_outputs_sum, eps_sum)


def random_choice_from_probs(softmax_inputs, temp=1., already_softmaxed=False):
  # temperature of infinity == take the max
  if temp == float('inf'):
    return np.argmax(softmax_inputs)
  if already_softmaxed:
    probs = softmax_inputs
    assert temp == 1
  else:
    probs = softmax(softmax_inputs, temp)
  r = random.random()
  cum_sum = 0.
  for i, p in enumerate(probs):
    cum_sum += p
    if cum_sum >= r: return i
  return 1  # return UNK?


def gen_stats(prob, normalizer=None):
  stats = {}
  stats['length'] = len(prob)
  stats['log_p'] = 0.0
  eps = 1e-12
  for p in prob:
    assert 0.0 <= p <= 1.0
    stats['log_p'] += np.log(max(eps, p))
  stats['log_p_word'] = stats['log_p'] / stats['length']
  stats['p'] = np.exp(stats['log_p'])
  stats['p_word'] = np.exp(stats['log_p_word'])
  try:
    stats['perplex'] = np.exp(-stats['log_p'])
  except OverflowError:
    stats['perplex'] = float('inf')
  try:
    stats['perplex_word'] = np.exp(-stats['log_p_word'])
  except OverflowError:
    stats['perplex_word'] = float('inf')
  if normalizer is not None:
    norm_stats = gen_stats(normalizer)
    stats['normed_perplex'] = stats['perplex'] / norm_stats['perplex']
    stats['normed_perplex_word'] = \
        stats['perplex_word'] / norm_stats['perplex_word']
  return stats
