# This code has been adapted from a version written by Jeff Donahue (jdonahue@eecs.berkeley.edu)

import os
import sys
import random
import copy
import argparse
from collections import defaultdict
import numpy as np
import h5py

sys.path.append('./Google_Refexp_toolbox/google_refexp_py_lib')
from refexp import Refexp
from refexp_eval import RefexpEvalComprehension
from common_utils import iou_bboxes
from hdf5_sequence_generator import SequenceGenerator, HDF5SequenceWriter
from shared_utils import UNK_IDENTIFIER, EOS_IDENTIFIER, EOQ_IDENTIFIER, get_encoded_line, image_feature_extractor
from experiment_settings import get_experiment_paths, get_experiment_config

random.seed(3)

class RefExpDataset(Refexp):

  def __init__(self, coco_instances_path, refexp_filename, image_root, split_name):
    Refexp.__init__(self, refexp_filename, coco_instances_path)
    self.image_refexp_pairs = []
    self.image_object_pairs = set()
    self.image_features = None
    self.imgs_with_errors = None
    self.image_feature_length = 0
    self.image_root = image_root
    self.img_to_refexps = defaultdict(list)

    sys.stdout.write("Collecting referring expressions... ")
    num_total = 0
    num_missing = 0
    image_ids = self.getImgIds()
    for image_id in image_ids:
      image_info = self.loadImgs(image_id)[0]
      image_path = '%s/%s' % (self.image_root, image_info['file_name'])
      if os.path.isfile(image_path):
        ann_ids = self.getAnnIds(image_id)
        anns = self.loadAnns(ann_ids)
        for ann in anns:
          object_id_list = [-1, ann['annotation_id']]  # -1 corresponds to the entire image
          refexp_anns = self.loadRefexps(ann['refexp_ids'])
          for refexp_ann in refexp_anns:
            refexp = refexp_ann['tokens'] + [EOS_IDENTIFIER]
            # image_refexp_pairs is used during extracting features for training
            self.image_refexp_pairs.append(((image_path, image_id), object_id_list, refexp))
            # img_to_refexps is used during testing
            self.img_to_refexps[image_id].append({'image_id':image_id, 'refexp':" ".join(refexp),
                                                  'object_id_list':object_id_list})
            # TODO: Remove redundancy in the above two data structures
            for object_id in object_id_list:
              if (image_id, object_id) not in self.image_object_pairs:
                self.image_object_pairs.add((image_id, object_id))
      else:
        num_missing += 1
        print 'Warning (#%d): image not found: %s' % (num_missing, image_path)

      num_total += 1
      if split_name == "train_debug" and num_total == 10:
        break

    print '%d/%d images missing' % (num_missing, num_total)

  def extract_image_object_features(self, features_filename, feature_layer='fc7', include_all_boxes=False,
                                    pre_existing_file=None):
    """
    :param features_filename: h5 file where CNN features are dumped.
    :param feature_layer: layer from which CNN features are extracted.
    :param include_all_boxes: if False, only groundtruth boxes of refexps are considered (baseline method)
                              if True, all groundtruth boxes are considered (max-margin, context methods)
    :param pre_existing_file: h5 file with previously extracted region features. Features that are already
                              extracted are skipped.
    """

    image_bbox_pairs = []
    image_infos = {}
    processed_pairs = set()

    if include_all_boxes:
      images_of_interest = set()
      for (image_id, obj_id) in self.image_object_pairs:
        images_of_interest.add(image_id)
      for image_id in images_of_interest:
        image = self.loadImgs(image_id)[0]
        image_infos[image_id] = image

        # Proposal boxes
        if 'region_candidates' in image:
          # Deep copy to prevent modifying groundtruth data
          anns = copy.deepcopy(image['region_candidates'])
          for ann in anns:
            ann['bbox'] = ann['bounding_box']
        else:
          anns = []

        # GT boxes
        if hasattr(self, 'coco'):
          anns += self.coco.imgToAnns[image_id]
        else:
          anns += self.imgToAnns[image_id]

        for ann in anns:
          if str((image_id, ann['bbox'])) not in processed_pairs:
            image_bbox_pairs.append((image_id, ann['bbox']))
            processed_pairs.add(str((image_id, ann['bbox'])))
    else:
      for (image_id, obj_id) in self.image_object_pairs:
        image = self.loadImgs(image_id)[0]
        image_infos[image_id] = image
        if obj_id == -1:
          img_wd = int(image['width'])
          img_ht = int(image['height'])
          bbox = [0,0,img_wd-1,img_ht-1]
          if str((image_id, bbox)) not in processed_pairs:
            image_bbox_pairs.append((image_id, bbox))
            processed_pairs.add(str((image_id, bbox)))
        else:
          bbox = self.loadAnns(obj_id)[0]['bbox']
          image_bbox_pairs.append((image_id, bbox))
          processed_pairs.add(str((image_id, bbox)))

    image_feature_extractor.extract_features_for_bboxes(self.image_root, image_infos, image_bbox_pairs,
                                                        features_filename, feature_layer, pre_existing_file)
    h5file = h5py.File(features_filename, 'r')
    self.image_features = h5file
    if 'imgs_with_errors' in h5file:
      self.imgs_with_errors = h5file['imgs_with_errors'][:]
    else:
      self.imgs_with_errors = []

    if feature_layer == 'fc7':
      self.image_feature_length = 4096
    elif feature_layer == 'fc8':
      self.image_feature_length = 1000
    else:
      raise AttributeError('Unknown feature layer %s' % feature_layer)


class GoogleRefExp(RefExpDataset):
  def __init__(self, split_name, experiment_paths):
    assert split_name in {'train', 'val', 'train_debug'}
    self.dataset_name = 'Google_RefExp_%s' % split_name
    self.dataset_name_wo_split = 'Google_RefExp'
    if split_name == "train_debug":
      refexp_filename = '%s/google_refexp_%s_201511_coco_aligned_mcg_umd.json' % (experiment_paths.google_refexp, "train")
    else:
      refexp_filename = '%s/google_refexp_%s_201511_coco_aligned_mcg_umd.json' % (experiment_paths.google_refexp, split_name)
    # All images in this dataset are originally from the COCO train split
    coco_instances_path = '%s/instances_%s.json' % (experiment_paths.coco_annotations, 'train2014')
    image_root = '%s/%s' % (experiment_paths.coco_images, 'train2014')
    self.evaluator = RefexpEvalComprehension(refexp_filename, coco_instances_path)

    RefExpDataset.__init__(self, coco_instances_path, refexp_filename, image_root, split_name)


class UNCRefExp(RefExpDataset):
  def __init__(self, split_name, experiment_paths):
    assert split_name in {'train', 'val', 'train_debug','test','testA','testB'}
    self.dataset_name = 'UNC_RefExp_%s' % split_name
    self.dataset_name_wo_split = 'UNC_RefExp'
    if split_name == "train_debug":
      refexp_filename = '%s/unc_refexp_%s_201602_coco_aligned.json' % (experiment_paths.unc_refexp, "train")
    else:
      refexp_filename = '%s/unc_refexp_%s_201602_coco_aligned.json' % (experiment_paths.unc_refexp, split_name)
    # All images in this dataset are originally from the COCO train split
    coco_instances_path = '%s/instances_%s.json' % (experiment_paths.coco_annotations, 'train2014')
    image_root = '%s/%s' % (experiment_paths.coco_images, 'train2014')
    self.evaluator = RefexpEvalComprehension(refexp_filename, coco_instances_path)

    RefExpDataset.__init__(self, coco_instances_path, refexp_filename, image_root, split_name)


class BaselineSequenceGenerator(SequenceGenerator):
  def __init__(self, experiment_paths, experiment_config, dataset, vocab=None, include_all_boxes=False):
    SequenceGenerator.__init__(self)
    self.exp_name = experiment_config.exp_name
    self.batch_num_streams = experiment_config.train.batch_size
    self.max_words = experiment_config.train.max_words
    self.pad = experiment_config.pad if hasattr(experiment_config, 'pad') else True
    self.truncate = experiment_config.truncate if hasattr(experiment_config, 'truncate') else True
    self.swap_axis_streams = frozenset(('timestep_input', 'timestep_cont', 'timestep_target'))
    self.index = 0
    self.num_resets = 0
    self.num_truncates = 0
    self.num_pads = 0
    self.num_outs = 0

    self.dataset = dataset
    self.image_refexp_pairs = self.dataset.image_refexp_pairs

    # Load image features
    if self.dataset.image_features is None:
      features_filename = "%s/COCO_region_features.h5" % experiment_paths.precomputed_image_features
      self.dataset.extract_image_object_features(features_filename, feature_layer='fc7',
                                                 include_all_boxes=include_all_boxes)

    # Load vocab
    if vocab is not None:
      self.vocabulary_inverted = vocab
      self.vocabulary = {}
      for index, word in enumerate(self.vocabulary_inverted):
        self.vocabulary[word] = index
    else:
      self.init_vocabulary()

    # make the number of image/refexp pairs a multiple of the buffer size
    # so each timestep of each batch is useful and we can align the images
    align = experiment_config.aligned if hasattr(experiment_config, 'aligned') else True
    if align:
      num_pairs = len(self.image_refexp_pairs)
      remainder = num_pairs % self.batch_num_streams
      if remainder > 0:
        num_needed = self.batch_num_streams - remainder
        for i in range(num_needed):
          choice = random.randint(0, num_pairs - 1)
          self.image_refexp_pairs.append(self.image_refexp_pairs[choice])
      assert len(self.image_refexp_pairs) % self.batch_num_streams == 0

    shuffle = experiment_config.shuffle if hasattr(experiment_config, 'shuffle') else True
    if shuffle:
      random.shuffle(self.image_refexp_pairs)

  def init_vocabulary(self, min_count=5):
    self.vocabulary = {}
    self.vocabulary_inverted = [UNK_IDENTIFIER, EOS_IDENTIFIER, EOQ_IDENTIFIER]
    for index, word in enumerate(self.vocabulary_inverted):
      self.vocabulary[word] = index

    words_to_count = {}
    exclude_set = {'..', '\'.', '.\\', '\".', '.\"'}
    for (_, _, sentence)in self.image_refexp_pairs:
      for word in sentence:
        try:
          word.decode('ascii')  # There are some words train that have issues
        except:
          continue
        word = word.strip()
        if (len(word) == 1 and word != "a" and not word.isdigit()) or word in exclude_set:
          continue
        if word not in self.vocabulary:
          if word not in words_to_count:
            words_to_count[word] = 0
          words_to_count[word] += 1

    # Sort words by count, then alphabetically
    words_by_count = sorted(words_to_count.keys(), key=lambda w: (-words_to_count[w], w))
    self.vocabulary_inverted += words_by_count
    prev_length = len(self.vocabulary)
    for index, word in enumerate(words_by_count):
      if word in words_to_count and words_to_count[word] < min_count:
        break
      self.vocabulary[word] = index + prev_length
    self.vocabulary_inverted = self.vocabulary_inverted[:len(self.vocabulary)]
    print 'Final vocabulary (restricted to words with counts of %d+) has %d words' % \
        (min_count, len(self.vocabulary))

  def dump_vocabulary(self, vocab_filename):
    print 'Dumping vocabulary to file: %s' % vocab_filename
    with open(vocab_filename, 'wb') as vocab_file:
      for word in self.vocabulary_inverted:
        vocab_file.write('%s\n' % word)

  def streams_exhausted(self):
    return self.num_resets > 0

  def next_line(self):
    num_lines = float(len(self.image_refexp_pairs))
    self.index += 1
    if self.index == 1 or self.index == num_lines or self.index % 10000 == 0:
      print 'Processed %d/%d (%0.2f%%) lines' % (self.index, num_lines,
                                                 100 * self.index / num_lines)
    if self.index == num_lines:
      self.index = 0
      self.num_resets += 1

  def swap_axis(self, stream_name):
    return True if stream_name in self.swap_axis_streams else False

  def get_streams(self):
    ((image_filename, image_id), object_id_list, line) = self.image_refexp_pairs[self.index]
    if image_id in self.dataset.imgs_with_errors:
      line = EOS_IDENTIFIER

    stream = get_encoded_line(line, self.vocabulary)
    # Assumes stream has EOS word at the end
    assert (stream[-1] == self.vocabulary[EOS_IDENTIFIER])
    stream = stream[:-1]
    filtered_stream = []
    for word in stream:
      if word != self.vocabulary[UNK_IDENTIFIER]:
        filtered_stream.append(word)
    stream = filtered_stream
    if self.truncate and len(stream) >= self.max_words:
      stream = stream[:self.max_words-1]
      self.num_truncates += 1
    pad = self.max_words - (len(stream) + 1) if self.pad else 0
    if pad > 0:
      self.num_pads += 1

    out = {}
    out['timestep_input']  = np.asarray([self.vocabulary[EOS_IDENTIFIER]] + stream + [-1] * pad, float)
    out['timestep_cont']   = np.asarray([0] + [1] * len(stream) + [0] * pad, float)
    out['timestep_target'] = np.asarray(stream + [self.vocabulary[EOS_IDENTIFIER]] + [-1] * pad, float)

    # Write image features to batch
    img_info = self.dataset.loadImgs(image_id)[0]
    img_wd = float(img_info['width'])
    img_ht = float(img_info['height'])

    out['fc7_img'] = self.dataset.image_features[str((image_id, [0, 0, int(img_wd - 1), int(img_ht - 1)]))][0]

    assert(object_id_list[0]==-1)
    object_id = object_id_list[1]
    bbox = self.dataset.loadAnns(object_id)[0]['bbox']
    out['fc7_obj'] = self.dataset.image_features[str((image_id, bbox))][0]
    bbox_area_ratio = (bbox[2] * bbox[3]) / (img_wd * img_ht)
    bbox_x1y1x2y2 = [bbox[0] / img_wd, bbox[1] / img_ht, (bbox[0] + bbox[2]) / img_wd, (bbox[1] + bbox[3]) / img_ht]
    bbox_features = bbox_x1y1x2y2 + [bbox_area_ratio]
    out['bbox_features'] = bbox_features

    self.num_outs += 1
    self.next_line()
    return out


class MaxMarginSequenceGenerator(BaselineSequenceGenerator):
  def __init__(self, experiment_paths, experiment_config, dataset, vocab=None):
    BaselineSequenceGenerator.__init__(self, experiment_paths, experiment_config,
                                       dataset, vocab=vocab, include_all_boxes=True)
    self.neg_proposal_source = experiment_config.train.neg_proposal_source \
                               if hasattr(experiment_config, 'train') else 'gt'
    if hasattr(experiment_config, 'train') and hasattr(experiment_config.train, 'max_num_negatives'):
      self.max_num_negatives = experiment_config.train.max_num_negatives
    else:
      self.max_num_negatives = 5
    self.swap_axis_streams = set()

  def get_streams(self):
    ((image_filename, image_id), object_id_list, line) = self.image_refexp_pairs[self.index]
    if image_id in self.dataset.imgs_with_errors:
      line = EOS_IDENTIFIER

    stream = get_encoded_line(line, self.vocabulary)
    # Assumes stream has EOS word at the end
    assert (stream[-1] == self.vocabulary[EOS_IDENTIFIER])
    stream = stream[:-1]
    filtered_stream = []
    for word in stream:
      if word != self.vocabulary[UNK_IDENTIFIER]:
        filtered_stream.append(word)
    stream = filtered_stream
    if self.truncate and len(stream) >= self.max_words:
      stream = stream[:self.max_words - 1]
      self.num_truncates += 1

    object_id = object_id_list[1]
    object_ann = self.dataset.loadAnns(object_id)[0]
    object_category = self.dataset.loadCats(object_ann['category_id'])[0]['name']
    object_bbox = self.dataset.loadAnns(object_id)[0]['bbox']
    neg_anns_of_same_category = []
    neg_anns_of_diff_category = []
    if self.neg_proposal_source != 'gt':
      image_info = self.dataset.loadImgs(image_id)[0]
      all_anns = image_info['region_candidates']
      for ann in all_anns:
        ann['bbox'] = ann['bounding_box']
        ann_box = ann['bbox']
        iou = iou_bboxes(ann_box, object_bbox)
        if iou < 0.5 and ann['predicted_object_name'] == object_category:
          neg_anns_of_same_category.append(ann)
        elif ann['predicted_object_name'] != object_category:
          neg_anns_of_diff_category.append(ann)
    else:
      if hasattr(self.dataset, 'coco'):
        all_anns = self.dataset.coco.imgToAnns[image_id]
      else:
        all_anns = self.dataset.imgToAnns[image_id]
      for ann in all_anns:
        if ann['id'] != object_id:
          if ann['category_id'] == object_ann['category_id']:
            neg_anns_of_same_category.append(ann)
          else:
            neg_anns_of_diff_category.append(ann)

    neg_anns = neg_anns_of_same_category  # Hard negatives
    if len(neg_anns) > self.max_num_negatives:
      rand_sample = sorted(random.sample(range(len(neg_anns)),self.max_num_negatives))
      neg_anns = [neg_anns[idx] for idx in rand_sample]
    elif len(neg_anns) < self.max_num_negatives:
      rand_sample = sorted(random.sample(range(len(neg_anns_of_diff_category)),
                                         min(self.max_num_negatives-len(neg_anns),len(neg_anns_of_diff_category))))
      neg_anns += [neg_anns_of_diff_category[idx] for idx in rand_sample]

      # If we are still running short of proposal negatives, sample from gt negatives
      if len(neg_anns) < self.max_num_negatives and self.neg_proposal_source != 'gt':
        if hasattr(self.dataset, 'coco'):
          all_anns = self.dataset.coco.imgToAnns[image_id]
        else:
          all_anns = self.dataset.imgToAnns[image_id]
        gt_neg_anns_of_same_category = []
        gt_neg_anns_of_diff_category = []
        for ann in all_anns:
          if ann['id'] != object_id:
            if ann['category_id'] == object_ann['category_id']:
              gt_neg_anns_of_same_category.append(ann)
            else:
              gt_neg_anns_of_diff_category.append(ann)
        rand_sample = sorted(random.sample(range(len(gt_neg_anns_of_diff_category)),
                                           min(self.max_num_negatives-len(neg_anns),
                                               len(gt_neg_anns_of_diff_category))))
        neg_anns += [gt_neg_anns_of_diff_category[idx] for idx in rand_sample]

    num_negatives = len(neg_anns)
    pad = self.max_words - (len(stream) + 1) if self.pad else 0
    if pad > 0:
      self.num_pads += 1

    out = {}
    timestep_input = np.asarray([[self.vocabulary[EOS_IDENTIFIER]] + stream + [-1] * pad], np.float16)
    out['timestep_input'] = np.tile(timestep_input.T, (1,self.max_num_negatives))
    timestep_cont = np.asarray([[0] + [1] * len(stream) + [0] * pad], np.float16)
    out['timestep_cont'] = np.tile(timestep_cont.T, (1,self.max_num_negatives))
    timestep_target = np.asarray([stream + [self.vocabulary[EOS_IDENTIFIER]] + [-1] * pad], np.float16)
    out['timestep_target'] = np.tile(timestep_target.T, (1,self.max_num_negatives))
    self.swap_axis_streams.add('timestep_input')
    self.swap_axis_streams.add('timestep_target')
    self.swap_axis_streams.add('timestep_cont')

    # Write image features to batch
    img_info = self.dataset.loadImgs(image_id)[0]
    img_wd = float(img_info['width'])
    img_ht = float(img_info['height'])
    assert(len(object_id_list) <= 2)
    fc7_img = self.dataset.image_features[str((image_id, [0, 0, int(img_wd - 1), int(img_ht - 1)]))][0]
    out['fc7_img'] = np.tile(fc7_img, (self.max_num_negatives,1))

    # Write object region features to batch
    fc7_obj = self.dataset.image_features[str((image_id, object_bbox))][0]
    out['fc7_obj'] = np.tile(fc7_obj, (self.max_num_negatives,1))

    bbox_area_ratio = (object_bbox[2] * object_bbox[3]) / (img_wd * img_ht)
    bbox_x1y1x2y2 = [object_bbox[0] / img_wd, object_bbox[1] / img_ht,
                     (object_bbox[0] + object_bbox[2]) / img_wd,
                     (object_bbox[1] + object_bbox[3]) / img_ht]
    bbox_features = bbox_x1y1x2y2 + [bbox_area_ratio]
    out['bbox_features'] = np.tile(bbox_features, (self.max_num_negatives,1))

    # Write negative features to batch
    negative_fc7_obj = np.zeros((self.max_num_negatives, self.dataset.image_feature_length),np.float16)
    negative_bbox_features = np.zeros((self.max_num_negatives, 5),np.float16)
    if len(neg_anns) > 0:
      other_bboxes = [ann['bbox'] for ann in neg_anns]
      for idx, other_bbox in enumerate(other_bboxes):
        other_bbox_area_ratio = (other_bbox[2] * other_bbox[3]) / (img_wd * img_ht)
        other_bbox_x1y1x2y2 = [other_bbox[0] / img_wd, other_bbox[1] / img_ht,
                               (other_bbox[0] + other_bbox[2]) / img_wd,
                               (other_bbox[1] + other_bbox[3]) / img_ht]
        other_bbox_features = other_bbox_x1y1x2y2 + [other_bbox_area_ratio]
        negative_fc7_obj[idx,:] = self.dataset.image_features[str((image_id, other_bbox))][0]
        negative_bbox_features[idx,:] = other_bbox_features
    out['negative_fc7_obj'] = negative_fc7_obj
    out['negative_bbox_features'] = negative_bbox_features
    pairwise_similarity = np.asarray([[0] * num_negatives + [-1] * (self.max_num_negatives-num_negatives)],
                                     np.float16)
    out['pairwise_similarity'] = np.tile(pairwise_similarity, (self.max_words,1))
    self.swap_axis_streams.add('pairwise_similarity')

    self.num_outs += 1
    self.next_line()
    return out


class MILContextSequenceGenerator(BaselineSequenceGenerator):
  def __init__(self, experiment_paths, experiment_config, dataset, vocab=None):
    BaselineSequenceGenerator.__init__(self,experiment_paths, experiment_config, dataset,
                                       vocab=vocab, include_all_boxes=True)
    self.neg_proposal_source = experiment_config.train.neg_proposal_source \
                               if hasattr(experiment_config, 'train') else 'gt'
    if hasattr(experiment_config, 'train') and hasattr(experiment_config.train, 'max_num_context'):
      self.max_num_context = experiment_config.train.max_num_context
    else:
      self.max_num_context = 5
    if hasattr(experiment_config, 'train') and hasattr(experiment_config.train, 'max_num_negatives'):
      self.max_num_negatives = experiment_config.train.max_num_negatives
    else:
      self.max_num_negatives = 5

    assert(self.max_num_context == self.max_num_negatives)
    self.swap_axis_streams = set()

  def get_streams(self):
    ((image_filename, image_id), object_id_list, line) = self.image_refexp_pairs[self.index]
    if image_id in self.dataset.imgs_with_errors:
      line = EOS_IDENTIFIER

    stream = get_encoded_line(line, self.vocabulary)
    # Assumes stream has EOS word at the end
    assert (stream[-1] == self.vocabulary[EOS_IDENTIFIER])
    stream = stream[:-1]
    filtered_stream = []
    for word in stream:
      if word != self.vocabulary[UNK_IDENTIFIER]:
        filtered_stream.append(word)
    stream = filtered_stream
    if self.truncate and len(stream) >= self.max_words:
      stream = stream[:self.max_words-1]
      self.num_truncates += 1

    object_id = object_id_list[1]
    object_ann = self.dataset.loadAnns(object_id)[0]
    object_category = self.dataset.loadCats(object_ann['category_id'])[0]['name']
    object_bbox = self.dataset.loadAnns(object_id)[0]['bbox']
    context_anns_of_same_category = []
    context_anns_of_diff_category = []
    if hasattr(self.dataset, 'coco'):
      all_anns = self.dataset.coco.imgToAnns[image_id]
    else:
      all_anns = self.dataset.imgToAnns[image_id]
    for ann in all_anns:
      if ann['id'] != object_id:
        if ann['category_id'] == object_ann['category_id']:
          context_anns_of_same_category.append(ann)
        else:
          context_anns_of_diff_category.append(ann)

    neg_anns_of_same_category = []
    neg_anns_of_diff_category = []
    if self.neg_proposal_source != 'gt':
      image_info = self.dataset.loadImgs(image_id)[0]
      all_anns = image_info['region_candidates']
      for ann in all_anns:
        ann['bbox'] = ann['bounding_box']
        ann_box = ann['bbox']
        iou = iou_bboxes(ann_box, object_bbox)
        if iou < 0.5 and ann['predicted_object_name'] == object_category:
          neg_anns_of_same_category.append(ann)
        elif ann['predicted_object_name'] != object_category:
          neg_anns_of_diff_category.append(ann)
    else:
      neg_anns_of_same_category = context_anns_of_same_category
      neg_anns_of_diff_category = context_anns_of_diff_category

    # subtract one because image is reserved as one context region
    if len(context_anns_of_same_category) > self.max_num_context-1:
      rand_sample = sorted(random.sample(range(len(context_anns_of_same_category)), self.max_num_context - 1))
      context_anns_of_same_category = [context_anns_of_same_category[idx] for idx in rand_sample]
    elif len(context_anns_of_same_category) < self.max_num_context-1:
      rand_sample = sorted(random.sample(range(len(context_anns_of_diff_category)),
                                         min(self.max_num_context - 1 - len(context_anns_of_same_category),
                                             len(context_anns_of_diff_category))))
      context_anns_of_same_category += [context_anns_of_diff_category[idx] for idx in rand_sample]

    if len(neg_anns_of_same_category) > self.max_num_negatives:
      rand_sample = sorted(random.sample(range(len(neg_anns_of_same_category)),self.max_num_negatives))
      neg_anns_of_same_category = [neg_anns_of_same_category[idx] for idx in rand_sample]
    elif len(neg_anns_of_same_category) < self.max_num_negatives:
      rand_sample = sorted(random.sample(range(len(neg_anns_of_diff_category)),
                                         min(self.max_num_negatives-len(neg_anns_of_same_category),
                                             len(neg_anns_of_diff_category))))
      neg_anns_of_same_category += [neg_anns_of_diff_category[idx] for idx in rand_sample]

      # If we are running short of proposal negatives, sample from gt negatives
      if len(neg_anns_of_same_category) < self.max_num_negatives and self.neg_proposal_source != 'gt':
        rand_sample = sorted(random.sample(range(len(context_anns_of_diff_category)),
                                           min(self.max_num_negatives-len(neg_anns_of_same_category),
                                               len(context_anns_of_diff_category))))
        neg_anns_of_same_category += [context_anns_of_diff_category[idx] for idx in rand_sample]

    pad = self.max_words - (len(stream) + 1) if self.pad else 0
    if pad > 0:
      self.num_pads += 1

    out = {}
    timestep_input = np.asarray([[self.vocabulary[EOS_IDENTIFIER]] + stream + [-1] * pad], np.float16)
    out['timestep_input'] = np.tile(timestep_input.T, (1,self.max_num_context))
    timestep_cont = np.asarray([[0] + [1] * len(stream) + [0] * pad], np.float16)
    out['timestep_cont'] = np.tile(timestep_cont.T, (1,self.max_num_context))
    timestep_target = np.asarray(stream + [self.vocabulary[EOS_IDENTIFIER]] + [-1] * pad, np.float16)
    out['timestep_target'] = timestep_target
    self.swap_axis_streams.add('timestep_input')
    self.swap_axis_streams.add('timestep_target')
    self.swap_axis_streams.add('timestep_cont')

    # Write image features to batch
    img_info = self.dataset.loadImgs(image_id)[0]
    img_wd = float(img_info['width'])
    img_ht = float(img_info['height'])
    assert(len(object_id_list) <= 2)
    fc7_img = self.dataset.image_features[str((image_id, [0, 0, int(img_wd - 1), int(img_ht - 1)]))][0]
    out['fc7_img'] = np.tile(fc7_img, (self.max_num_context, 1))
    img_bbox_features = np.zeros((self.max_num_context, 5), np.float16)
    img_bbox_features[:] = [0,0,1,1,1]
    out['img_bbox_features'] = img_bbox_features

    # Write object region features to batch
    object_bbox = self.dataset.loadAnns(object_id)[0]['bbox']
    fc7_obj = self.dataset.image_features[str((image_id, object_bbox))][0]
    out['fc7_obj'] = np.tile(fc7_obj, (self.max_num_context, 1))

    bbox_area_ratio = (object_bbox[2] * object_bbox[3]) / (img_wd * img_ht)
    bbox_x1y1x2y2 = [object_bbox[0] / img_wd, object_bbox[1] / img_ht,
                     (object_bbox[0] + object_bbox[2]) / img_wd, (object_bbox[1] + object_bbox[3]) / img_ht]
    bbox_features = bbox_x1y1x2y2 + [bbox_area_ratio]
    out['bbox_features'] = np.tile(bbox_features, (self.max_num_context, 1))

    # Write context features to batch
    context_fc7 = np.tile(fc7_img, (self.max_num_context, 1))
    context_bbox_features = np.zeros((self.max_num_context, 5), np.float16)
    context_bbox_features[:] = [0,0,1,1,1]
    if len(context_anns_of_same_category) > 0:
      other_bboxes = [ann['bbox'] for ann in context_anns_of_same_category]
      for idx, other_bbox in enumerate(other_bboxes):
        other_bbox_area_ratio = (other_bbox[2] * other_bbox[3]) / (img_wd * img_ht)
        other_bbox_x1y1x2y2 = [other_bbox[0] / img_wd, other_bbox[1] / img_ht,
                               (other_bbox[0] + other_bbox[2]) / img_wd, (other_bbox[1] + other_bbox[3]) / img_ht]
        other_bbox_features = other_bbox_x1y1x2y2 + [other_bbox_area_ratio]
        context_fc7[idx,:] = self.dataset.image_features[str((image_id, other_bbox))][0]
        context_bbox_features[idx,:] = other_bbox_features
    out['context_fc7'] = context_fc7
    out['context_bbox_features'] = context_bbox_features

    # Write negative features to batch
    negative_fc7 = np.zeros((self.max_num_negatives, self.dataset.image_feature_length),np.float16)
    negative_bbox_features = np.zeros((self.max_num_negatives, 5),np.float16)
    if len(neg_anns_of_same_category) > 0:
      other_bboxes = [ann['bbox'] for ann in neg_anns_of_same_category]
      for idx, other_bbox in enumerate(other_bboxes):
        other_bbox_area_ratio = (other_bbox[2] * other_bbox[3]) / (img_wd * img_ht)
        other_bbox_x1y1x2y2 = [other_bbox[0] / img_wd, other_bbox[1] / img_ht,
                               (other_bbox[0] + other_bbox[2]) / img_wd, (other_bbox[1] + other_bbox[3]) / img_ht]
        other_bbox_features = other_bbox_x1y1x2y2 + [other_bbox_area_ratio]
        negative_fc7[idx,:] = self.dataset.image_features[str((image_id, other_bbox))][0]
        negative_bbox_features[idx,:] = other_bbox_features
    out['negative_fc7'] = negative_fc7
    out['negative_bbox_features'] = negative_bbox_features

    pairwise_similarity = np.asarray([[0] * self.max_num_negatives], np.float16)
    out['pairwise_similarity'] = np.tile(pairwise_similarity, (self.max_words,1))
    self.swap_axis_streams.add('pairwise_similarity')

    self.num_outs += 1
    self.next_line()
    return out


def process_dataset(experiment_paths, experiment_config):
  vocab = None
  split_names = ['train', 'val']
  if hasattr(experiment_config,'debug') and experiment_config.debug:
    split_names = ['train_debug']
  for split_name in split_names:
    if split_name == 'val' and vocab is None:
      raise StandardError('Need a vocabulary constructed from train split')

    #TODO: Does saving and reading the dataset from a pickle file save time?
    if experiment_config.dataset == 'Google_RefExp':
      dataset = GoogleRefExp(split_name, experiment_paths)
    elif experiment_config.dataset == 'UNC_RefExp':
      dataset = UNCRefExp(split_name, experiment_paths)
    else:
      raise Exception('Unknown dataset: %s' % experiment_config.dataset)

    print "Processing dataset %s" % dataset.dataset_name
    if experiment_config.exp_name == 'baseline':
      sg = BaselineSequenceGenerator(experiment_paths, experiment_config, dataset, vocab=vocab)
    elif experiment_config.exp_name.startswith('max_margin'):
      sg = MaxMarginSequenceGenerator(experiment_paths, experiment_config, dataset, vocab=vocab)
    elif experiment_config.exp_name.startswith('mil_context'):
      sg = MILContextSequenceGenerator(experiment_paths, experiment_config, dataset, vocab=vocab)
    else:
      raise StandardError("Unknown experiment name %s" % experiment_config.exp_name)
    output_dir = "%s/buffer_%d/%s_%s" % (experiment_paths.h5_data, experiment_config.train.batch_size,
                                         dataset.dataset_name,experiment_config.train.tag)
    writer = HDF5SequenceWriter(sg, output_dir=output_dir)
    writer.write_to_exhaustion()
    writer.write_filelists()
    sg.dataset.image_features.close()

    if vocab is None:
      sg.dump_vocabulary(experiment_config.vocab_file)
      vocab = sg.vocabulary_inverted

    print 'Padded %d/%d sequences; truncated %d/%d sequences' % \
          (sg.num_pads, sg.num_outs, sg.num_truncates, sg.num_outs)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--coco_path', required=True, help='Path to MSCOCO dataset')
  parser.add_argument('--dataset', required=True, help='Name of the dataset. [Google_RefExp|UNC_RefExp]')
  parser.add_argument('--exp_name', required=True, help='Type of model. [baseline|max-margin|mil_context]')
  args = parser.parse_args()
  exp_paths = get_experiment_paths(args.coco_path)
  exp_config = get_experiment_config(exp_paths,args.dataset,args.exp_name,mode='train')
  process_dataset(exp_paths, exp_config)