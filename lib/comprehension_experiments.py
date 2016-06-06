import sys
import random
import json
import argparse
import h5py
import ipdb
import matplotlib.pyplot as plt
plt.ion()
import matplotlib.image as mpimg
from collections import defaultdict, OrderedDict
import numpy as np

from language_model import LanguageModel, MILContextLanguageModel, gen_stats
from shared_utils import UNK_IDENTIFIER, image_feature_extractor, get_encoded_line
from experiment_settings import get_experiment_paths, get_experiment_config
from process_dataset import GoogleRefExp, UNCRefExp

class ComprehensionExperiment:
  def __init__(self, language_model, dataset, image_ids=None):
    self.lang_model = language_model
    self.dataset = dataset
    if image_ids:
      self.images = image_ids
    else:
      self.images = self.dataset.getImgIds()

  def extract_image_features(self, experiment_paths, proposal_source, output_h5_file):
    img_bbox_pairs = []
    image_infos = {}
    processed_pairs = set()
    for image_id in self.images:
      image = self.dataset.loadImgs(image_id)[0]
      image_infos[image_id] = image
      if proposal_source != 'gt':
        bboxes = [cand['bounding_box'] for cand in image['region_candidates']]
      else:
        anns = self.dataset.coco.imgToAnns[image_id]
        bboxes = [ann['bbox'] for ann in anns]
      if len(bboxes) == 0:
        continue
      img_wd = int(image['width'])
      img_ht = int(image['height'])
      bboxes += [[0,0,img_wd-1,img_ht-1]]
      for bbox in bboxes:
        if str((image_id, bbox)) not in processed_pairs:
          processed_pairs.add(str((image_id, bbox)))
          img_bbox_pairs.append((image_id, bbox))

    image_feature_extractor.extract_features_for_bboxes(self.dataset.image_root, image_infos,
                                                        img_bbox_pairs, output_h5_file, feature_layer='fc7')

  def comprehension_experiment(self, experiment_paths, proposal_source='gt', visualize=False, eval_method=None):
    output_h5_file = '%s/COCO_region_features.h5' % experiment_paths.precomputed_image_features
    self.extract_image_features(experiment_paths, proposal_source, output_h5_file)
    h5file = h5py.File(output_h5_file, 'r')

    num_images = len(self.images)
    random.seed()
    random.shuffle(self.images)
    results = []
    for (i,image_id) in enumerate(self.images):
      image = self.dataset.loadImgs(image_id)[0]
      if proposal_source != 'gt':
        bboxes = [cand['bounding_box'] for cand in image['region_candidates']]
      else:
        obj_anns = self.dataset.coco.imgToAnns[image_id]
        bboxes = [ann['bbox'] for ann in obj_anns]

      if len(bboxes) == 0:
        print("No region candidates for %d" % image_id)
        anns = self.dataset.img_to_refexps[image_id]
        for ann in anns:
          gt_obj = ann['object_id_list'][1] if len(ann['object_id_list']) == 2 else -1
          result = {'annotation_id':gt_obj, 'predicted_bounding_boxes':[], 'refexp':ann['refexp'][0]}
          results.append(result)
        continue

      # Object region features
      for obj_i in range(len(bboxes)):
        feats = h5file[str((image_id,bboxes[obj_i]))][:]
        if obj_i == 0:
          obj_feats = feats
        else:
          obj_feats = np.vstack((obj_feats, feats))
      # Image region features
      img_wd = int(image['width'])
      img_ht = int(image['height'])
      img_feats = h5file[str((image_id,[0,0,img_wd-1,img_ht-1]))][:]
      img_feats = np.tile(img_feats,(len(obj_feats),1))
      # Bounding box features
      bbox_features = []
      for bbox in bboxes:
        img_wd = float(img_wd)
        img_ht = float(img_ht)
        bbox_area_ratio = (bbox[2]*bbox[3])/(img_wd*img_ht)
        bbox_x1y1x2y2 = [bbox[0]/img_wd, bbox[1]/img_ht,
                         min(1., (bbox[0]+bbox[2])/img_wd), min(1., (bbox[1]+bbox[3])/img_ht)]
        bbox_features.append(bbox_x1y1x2y2 + [bbox_area_ratio])

      anns = self.dataset.img_to_refexps[image_id]
      for ann in anns:
        prefix_words_unfiltered = get_encoded_line(ann['refexp'], self.lang_model.vocab)
        prefix_words = []
        for word in prefix_words_unfiltered:
          if word != self.lang_model.vocab[UNK_IDENTIFIER]:
            prefix_words.append(word)
        prefix_words = [prefix_words] * len(bboxes)
        output_captions, output_probs = self.lang_model.sample_captions(obj_feats, img_feats, bbox_features,
                                                                        prefix_words=prefix_words)
        stats = [gen_stats(output_prob) for output_prob in output_probs]
        stats = [stat['log_p_word'] for stat in stats]
        (sort_keys, sorted_stats) = zip(*sorted(enumerate(stats), key=lambda x:-x[1]))
        top_k = 10 if len(sort_keys) > 10 else len(sort_keys)
        top_bboxes = [bboxes[k] for k in sort_keys[:top_k]]

        gt_obj = ann['object_id_list'][1] if len(ann['object_id_list']) == 2 else -1
        result = {'annotation_id':gt_obj, 'predicted_bounding_boxes':top_bboxes, 'refexp':ann['refexp']}

        if visualize:
          img_filename = '%s/%s' % (self.dataset.image_root, self.dataset.loadImgs(image_id)[0]['file_name'])
          im = mpimg.imread(img_filename)
          plt.cla()
          plt.imshow(im)
          plt.axis('off')
          plt.title(ann['refexp'])

          if gt_obj != -1:
            gt_box = self.dataset.coco.loadAnns(gt_obj)[0]['bbox']
            plt.gca().add_patch(plt.Rectangle((gt_box[0], gt_box[1]),gt_box[2], gt_box[3],
                                              fill=False, edgecolor='g', linewidth=3))

          top_box = top_bboxes[0]
          plt.gca().add_patch(plt.Rectangle((top_box[0], top_box[1]),top_box[2], top_box[3],
                                            fill=False, edgecolor='r', linewidth=3))
          #top_box_score = stats[bboxes.index(top_box)]
          #plt.text(top_box[0], top_box[1], str(top_box_score), fontsize=12, bbox=dict(facecolor='red', alpha=1))

          ipdb.set_trace()

        results.append(result)

      sys.stdout.write("\rDone with %d/%d images" % (i+1,num_images))
      sys.stdout.flush()

    sys.stdout.write("\n")
    h5file.close()
    return results


class MILContextComprehensionExperiment(ComprehensionExperiment):
  def __init__(self, language_model, dataset, image_ids=None):
    ComprehensionExperiment.__init__(self, language_model, dataset, image_ids=image_ids)

  def comprehension_experiment(self, experiment_paths, proposal_source='gt', visualize=False, eval_method=None):
    output_h5_file = '%s/COCO_region_features.h5' % experiment_paths.precomputed_image_features
    self.extract_image_features(experiment_paths, proposal_source, output_h5_file)
    h5file = h5py.File(output_h5_file, 'r')

    if eval_method is None:
      eval_methods = ['noisy_or', 'max', 'image_context_only']
    else:
      eval_methods = [eval_method]

    results = defaultdict(list)
    num_images = len(self.images)
    random.seed()
    random.shuffle(self.images)
    for (i, image_id) in enumerate(self.images):
      image = self.dataset.loadImgs(image_id)[0]
      if proposal_source != 'gt':
        bboxes = [cand['bounding_box'] for cand in image['region_candidates']]
      else:
        anns = self.dataset.coco.imgToAnns[image_id]
        bboxes = [ann['bbox'] for ann in anns]

      if len(bboxes) == 0:
        print("No region candidates for %d" % image_id)
        anns = self.dataset.img_to_refexps[image_id]
        for ann in anns:
          gt_obj = ann['object_id_list'][1] if len(ann['object_id_list']) == 2 else -1
          result = {'annotation_id':gt_obj, 'predicted_bounding_boxes':[], 'refexp':ann['refexp']}
          for method in eval_methods:
            results[method].append(result)
        continue

      # Image region features
      batch_size = len(bboxes)
      img_wd = int(image['width'])
      img_ht = int(image['height'])
      fc7_img = h5file[str((image_id,[0,0,img_wd-1,img_ht-1]))][:]

      img_wd = float(img_wd)
      img_ht = float(img_ht)
      image_feature_length = len(fc7_img[0])
      # Any change to context_length value will also require a change in the deploy prototxt
      context_length = 10
      fc7_obj = np.zeros((batch_size,context_length,image_feature_length))
      context_fc7 = np.tile(fc7_img,(batch_size,context_length,1))
      bbox_features = np.zeros((batch_size,context_length,5))
      context_bbox_features = np.zeros((batch_size,context_length, 5),np.float16)

      context_bboxes = []
      for (bbox_idx, bbox) in enumerate(bboxes):
        # Object region features
        fc7_obj[bbox_idx,:] = h5file[str((image_id,bbox))][:]

        # Bounding box features
        bbox_area_ratio = (bbox[2]*bbox[3])/(img_wd*img_ht)
        bbox_x1y1x2y2 = [bbox[0]/img_wd, bbox[1]/img_ht,
                         min(1., (bbox[0]+bbox[2])/img_wd), min(1., (bbox[1]+bbox[3])/img_ht)]
        obj_bbox_features = bbox_x1y1x2y2 + [bbox_area_ratio]
        bbox_features[bbox_idx,:] = obj_bbox_features
        context_bbox_features[bbox_idx,:] = [0,0,1,1,1]

        # Context features
        other_bboxes = list(bboxes)  # make a copy
        other_bboxes.remove(bbox)

        if len(other_bboxes) > context_length-1:
          rand_sample = sorted(random.sample(range(len(other_bboxes)),context_length-1))
          other_bboxes = [other_bboxes[idx] for idx in rand_sample]

        context_bboxes.append(other_bboxes)

        for (other_bbox_idx, other_bbox) in enumerate(other_bboxes):
          other_bbox_area_ratio = (other_bbox[2] * other_bbox[3]) / (img_wd * img_ht)
          other_bbox_x1y1x2y2 = [other_bbox[0] / img_wd, other_bbox[1] / img_ht,
                                 (other_bbox[0] + other_bbox[2]) / img_wd, (other_bbox[1] + other_bbox[3]) / img_ht]
          other_bbox_features = other_bbox_x1y1x2y2 + [other_bbox_area_ratio]
          feats = h5file[str((image_id,other_bbox))][:]
          context_fc7[bbox_idx,other_bbox_idx,:] = feats
          context_bbox_features[bbox_idx,other_bbox_idx,:] = other_bbox_features

      for elem in context_bboxes:
        elem.append([0,0,img_wd-1,img_ht-1])

      anns = self.dataset.img_to_refexps[image_id]
      for ann in anns:
        prefix_words_unfiltered = get_encoded_line(ann['refexp'], self.lang_model.vocab)
        prefix_words = []
        for word in prefix_words_unfiltered:
          if word != self.lang_model.vocab[UNK_IDENTIFIER]:
            prefix_words.append(word)
        prefix_words = [prefix_words] * batch_size
        output_captions, output_probs, \
        output_all_probs = self.lang_model.sample_captions_with_context(fc7_obj, bbox_features,
                                                                        context_fc7, context_bbox_features,
                                                                        prefix_words=prefix_words)
        all_stats = [gen_stats(output_prob) for output_prob in output_all_probs]
        all_stats_p_word = [stat['p_word'] for stat in all_stats]
        all_stats_p_word = np.reshape(all_stats_p_word, (batch_size, context_length))

        for method in eval_methods:
          if method == 'noisy_or':
            num_context_objs = min(context_length-1,len(bboxes)-1)
            sort_all_stats_p_word = -np.sort(-all_stats_p_word[:,0:num_context_objs])
            top_all_stats_p_word = np.hstack((sort_all_stats_p_word,all_stats_p_word[:,-1:]))
            stats = (1 - np.product(1-top_all_stats_p_word,axis=1))
          elif method == 'image_context_only':
            stats = all_stats_p_word[:,-1]
          elif method == 'max':
            stats = np.max(all_stats_p_word,axis=1)
          else:
            raise StandardError("Unknown eval method %s" % method)

          (sort_keys, sorted_stats) = zip(*sorted(enumerate(stats), key=lambda x:-x[1]))
          top_k = 10 if len(sort_keys) > 10 else len(sort_keys)
          top_bboxes = [bboxes[k] for k in sort_keys[:top_k]]
          gt_obj = ann['object_id_list'][1] if len(ann['object_id_list']) == 2 else -1
          result = {'annotation_id':gt_obj, 'predicted_bounding_boxes':top_bboxes, 'refexp':ann['refexp']}
          results[method].append(result)

          gt_box = self.dataset.coco.loadAnns(gt_obj)[0]['bbox']
          if method == 'noisy_or':
            noisy_or_top_box = top_bboxes[0]
          elif method == "image_context_only":
            image_top_bbox = top_bboxes[0]

        if visualize:
          print "Image id: %d" % image_id
          img_filename = '%s/%s' % (self.dataset.image_root, self.dataset.loadImgs(image_id)[0]['file_name'])
          im = mpimg.imread(img_filename)

          if noisy_or_top_box:
            plt.figure(1)
            plt.cla()
            plt.imshow(im)
            plt.title(ann['refexp'])
            top_box = noisy_or_top_box
            top_box_ind = bboxes.index(top_box)
            plt.gca().add_patch(plt.Rectangle((top_box[0], top_box[1]),top_box[2], top_box[3],
                                              fill=False, edgecolor='b', linewidth=6))
            top_context_box_ind = np.argmax(all_stats_p_word[top_box_ind])
            top_context_box = context_bboxes[top_box_ind][top_context_box_ind]
            plt.gca().add_patch(plt.Rectangle((top_context_box[0], top_context_box[1]),top_context_box[2],
                                              top_context_box[3], fill=False, edgecolor='b', linewidth=6,
                                              linestyle='dashed'))
            plt.axis('off')

          if image_top_bbox:
            plt.figure(2)
            plt.cla()
            plt.imshow(im)
            plt.title(ann['refexp'])
            top_box = image_top_bbox
            plt.gca().add_patch(plt.Rectangle((top_box[0], top_box[1]),top_box[2], top_box[3],
                                              fill=False, edgecolor='b', linewidth=6))
            plt.axis('off')

          plt.figure(3)
          plt.cla()
          plt.imshow(im)
          plt.title(ann['refexp'])
          plt.gca().add_patch(plt.Rectangle((gt_box[0], gt_box[1]),gt_box[2], gt_box[3],
                                            fill=False, edgecolor='g', linewidth=6))
          plt.axis('off')

          while True:
            sys.stdout.write('Do you want to save? (y/n): ')
            choice = raw_input().lower()
            if choice.startswith('y'):
              plt.figure(1)
              plt.savefig('%s/%d_nor.png' % (experiment_paths.coco_path, image_id),bbox_inches='tight')
              plt.figure(2)
              plt.savefig('%s/%d_image_context.png' % (experiment_paths.coco_path, image_id),bbox_inches='tight')
              plt.figure(3)
              plt.savefig('%s/%d_gt.png' % (experiment_paths.coco_path, image_id),bbox_inches='tight')
              break
            elif choice.startswith('n'):
              break

          ipdb.set_trace()

      sys.stdout.write("\rDone with %d/%d images" % (i+1,num_images))
      sys.stdout.flush()

    sys.stdout.write("\n")
    h5file.close()
    return results


def run_comprehension_experiment(dataset, experiment_paths, experiment_config, image_ids=None):
  if experiment_config.exp_name == 'baseline' or experiment_config.exp_name.startswith('max_margin'):
    captioner = LanguageModel(experiment_config.test.lstm_model_file, experiment_config.test.lstm_net_file,
                              experiment_config.vocab_file, device_id=0)
  elif experiment_config.exp_name.startswith('mil_context'):
    captioner = MILContextLanguageModel(experiment_config.test.lstm_model_file, experiment_config.test.lstm_net_file,
                                        experiment_config.vocab_file, device_id=0)
  else:
      raise StandardError("Unknown experiment name: %s" % experiment_config.exp_name)

  if experiment_config.exp_name == 'baseline' or experiment_config.exp_name.startswith('max_margin'):
    experimenter = ComprehensionExperiment(captioner, dataset, image_ids=image_ids)
  elif experiment_config.exp_name.startswith('mil_context'):
    experimenter = MILContextComprehensionExperiment(captioner, dataset, image_ids=image_ids)
  else:
    raise StandardError("Unknown experiment name: %s" % experiment_config.exp_name)

  results = experimenter.comprehension_experiment(experiment_paths, proposal_source=experiment_config.test.proposal_source,
                                                  visualize=experiment_config.test.visualize)

  if isinstance(results,dict):
    for method in results:
      print "Results for method: %s" % method
      results_filename = '%s/%s_%s_%s_results.json' % (experiment_paths.retrieval_results, dataset.dataset_name,
                                                       experiment_config.test.tag, method)
      with open(results_filename,'w') as f: json.dump(results[method], f)
  else:
    results_filename = '%s/%s_%s_results.json' % (experiment_paths.retrieval_results, dataset.dataset_name,
                                                           experiment_config.test.tag)
    with open(results_filename,'w') as f: json.dump(results, f)


def get_results(dataset, exp_name, experiment_paths, experiment_config):
  results = OrderedDict()
  if exp_name.startswith('mil_context'):
    for method in ['noisy_or', 'max', 'image_context_only']:
      results_filename = '%s/%s_%s_%s_results.json' % (experiment_paths.retrieval_results, dataset.dataset_name,
                                                       experiment_config.test.tag, method)
      (prec, eval_results) = dataset.evaluator.evaluate(results_filename, flag_ignore_non_existed_object=True,
                                                        flag_ignore_non_existed_gt_refexp=True)
      results['%s_%s' % (exp_name, method)] = round(prec,3)
  else:
    results_filename = '%s/%s_%s_results.json' % (experiment_paths.retrieval_results, dataset.dataset_name,
                                                experiment_config.test.tag)
    (prec, eval_results) = dataset.evaluator.evaluate(results_filename, flag_ignore_non_existed_object=True,
                                                    flag_ignore_non_existed_gt_refexp=True)
    results[exp_name] = round(prec,3)

  return results


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--coco_path', required=True, help='Path to MSCOCO dataset')
  parser.add_argument('--dataset', required=True, help='Name of the dataset. [Google_RefExp|UNC_RefExp]')
  parser.add_argument('--exp_name', required=True, nargs='+',
                      help='Type of model. [baseline|max-margin|mil_context_withNegMargin|mil_context_withPosNegMargin|all]')
  parser.add_argument('--split_name', required=True, help='Partition to test on')
  parser.add_argument('--proposal_source', required=True, help='Test time proposal source [gt|mcg]')
  parser.add_argument('--visualize', action="store_true", default=False, help='Display comprehension results')

  args = parser.parse_args()
  exp_paths = get_experiment_paths(args.coco_path)
  if args.dataset == 'Google_RefExp':
    dataset = GoogleRefExp(args.split_name, exp_paths)
  elif args.dataset == 'UNC_RefExp':
    dataset = UNCRefExp(args.split_name, exp_paths)
  else:
    raise Exception('Unknown dataset: %s' % args.dataset)

  if 'all' in args.exp_name:
    args.exp_name = ['baseline', 'max_margin', 'mil_context_withNegMargin', 'mil_context_withPosNegMargin']

  # Run all experiments first
  for exp_name in args.exp_name:
    exp_config = get_experiment_config(exp_paths,args.dataset,exp_name,mode='test',
                                       test_proposal_source=args.proposal_source, test_visualize=args.visualize)
    run_comprehension_experiment(dataset, exp_paths, exp_config)

  # Print results for all experiments
  all_exp_results = OrderedDict()
  for exp_name in args.exp_name:
    exp_config = get_experiment_config(exp_paths,args.dataset,exp_name,mode='test',
                                       test_proposal_source=args.proposal_source, test_visualize=args.visualize)
    exp_results = get_results(dataset, exp_name, exp_paths, exp_config)
    all_exp_results.update(exp_results)

  print all_exp_results
