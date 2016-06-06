import os
import argparse
from easydict import EasyDict as edict

def get_experiment_paths(coco_path):
  paths = edict()
  # COCO paths
  paths.coco_path = coco_path
  paths.coco_annotations = '%s/annotations' % coco_path
  paths.coco_images = '%s/images' % coco_path
  paths.coco_proposals = '%s/proposals/MCG/mat' % coco_path

  # Google Refexp path
  paths.google_refexp = '%s/google_refexp' % coco_path

  # UNC Refexp path
  paths.unc_refexp = '%s/unc_refexp' % coco_path

  # Cache dir
  cache_dir = '%s/cache_dir' % coco_path
  paths.h5_data = '%s/h5_data' % cache_dir
  paths.models = '%s/models' % cache_dir
  paths.pickles = '%s/pickles' % cache_dir
  paths.precomputed_image_features = '%s/precomputed_image_features' % cache_dir
  paths.retrieval_results = '%s/comprehension_results' % cache_dir
  for p in [paths.h5_data, paths.models, paths.pickles, paths.precomputed_image_features, paths.retrieval_results]:
    if not os.path.exists(p):
      os.makedirs(p)

  return paths


def get_experiment_config(experiment_paths,dataset=None,exp_name=None,mode='train',
                          test_proposal_source=None, test_visualize=False):
  config = edict()

  assert(dataset in ['Google_RefExp', 'UNC_RefExp'])
  config.dataset = 'Google_RefExp' if dataset is None else dataset
  assert((exp_name in ['baseline', 'max_margin']) or exp_name.startswith('mil_context'))
  config.exp_name = 'baseline' if exp_name is None else exp_name
  config.vocab_file = '%s/%s_vocabulary.txt' % (experiment_paths.h5_data, config.dataset)
  config.debug = False

  assert(mode in ['train', 'test'])
  # Training parameters
  config.train = edict()
  config.train.batch_size = 16
  config.train.max_words = 20 if config.dataset.startswith('Google_RefExp') else 10
  if config.exp_name == 'baseline':
    config.train.neg_proposal_source = 'gt'
  else:
    config.train.neg_proposal_source = 'mcg'
  config.train.tag = "%s_%s_%d" % (config.exp_name, config.train.neg_proposal_source, config.train.max_words)

  # Testing parameters
  if mode == 'test':
    config.test = edict()
    config.test.proposal_source = 'gt' if test_proposal_source is None else test_proposal_source
    config.test.tag = "_".join([config.exp_name, config.test.proposal_source])
    config.test.visualize = test_visualize
    if config.dataset == 'Google_RefExp':
      test_model_iteration_num = {'baseline_gt': 240000, 'max_margin_mcg': 160000,
                                  'mil_context_withNegMargin_mcg': 140000, 'mil_context_withPosNegMargin_mcg': 110000}
    elif config.dataset == 'UNC_RefExp':
      test_model_iteration_num = {'baseline_gt': 220000, 'max_margin_mcg': 240000,
                                  'mil_context_withNegMargin_mcg': 160000, 'mil_context_withPosNegMargin_mcg': 110000}
    model_tag = "%s_%s" % (config.exp_name, config.train.neg_proposal_source)
    config.test.iter = test_model_iteration_num[model_tag]
    config.test.lstm_net_file = './proto_files/%s/%s.%s.deploy.prototxt' % (config.dataset.lower(),
                                                                            config.dataset.lower(), config.exp_name)
    config.test.lstm_model_file = '%s/%s_%s/%s.%s_iter_%d.caffemodel' % (experiment_paths.models,
                                                                         config.dataset, model_tag,
                                                                         config.dataset.lower(), model_tag,
                                                                         config.test.iter)

  return config


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--coco_path', required=True, help='Path to MSCOCO dataset')

  args = parser.parse_args()
  print "Creating cache directories for experiments"
  get_experiment_paths(args.coco_path)
