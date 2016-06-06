import os
import sys
import h5py
import numpy as np

sys.path.append('./caffe/python')
import caffe

UNK_IDENTIFIER = '<UNK>'
EOS_IDENTIFIER = '<EOS>'
EOQ_IDENTIFIER = '<EOQ>'


def get_encoded_line(sentence, vocabulary):
  stream = []
  if isinstance(sentence, unicode):
    sentence = str(sentence).split()
  elif isinstance(sentence, str):
    sentence = sentence.split()
  for word in sentence:
    word = word.strip()
    if word in vocabulary:
      stream.append(vocabulary[word])
    else:  # unknown word; append UNK
      stream.append(vocabulary[UNK_IDENTIFIER])
  return stream


class ImageFeatureExtractor:
  def __init__(self, image_net='caffenet'):
    self.initialized = False
    self.image_net = image_net
    self.net = None
    self.transformer = None
    self.BATCH_SIZE = 1

  def init(self):
    image_net = self.image_net
    caffe.set_device(0)
    caffe.set_mode_gpu()
    if image_net == 'caffenet':
      convnet_proto = './caffe/models/bvlc_reference_caffenet/deploy.prototxt'
      convnet_model = './caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
    elif image_net == 'vggnet':
      convnet_proto = './caffe/models/vggnet/VGG_ILSVRC_16_layers_deploy.prototxt'
      convnet_model = './caffe/models/vggnet/VGG_ILSVRC_16_layers.caffemodel'
    else:
      raise StandardError('Unknown CNN %s' % image_net)

    self.net = caffe.Net(convnet_proto, convnet_model, caffe.TEST)

    if image_net == 'caffenet':
      self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
      self.transformer.set_transpose('data', (2, 0, 1))
      self.transformer.set_mean('data', np.array([104, 117, 123]))
      self.transformer.set_raw_scale('data', 255)
      self.transformer.set_channel_swap('data', (2, 1, 0))
      self.BATCH_SIZE = 100
      self.net.blobs['data'].reshape(self.BATCH_SIZE, 3, 227, 227)
    elif image_net == 'vggnet':
      self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
      self.transformer.set_transpose('data', (2, 0, 1))
      self.transformer.set_mean('data', np.array([103.939, 116.779, 123.68]))
      self.transformer.set_raw_scale('data', 255)
      self.transformer.set_channel_swap('data', (2, 1, 0))
      self.BATCH_SIZE = 100
      self.net.blobs['data'].reshape(self.BATCH_SIZE, 3, 224, 224)

    self.image_net = image_net
    self.initialized = True
    print "Done initializing image feature extractor"


  def extract_features_for_bboxes(self, image_root, image_infos, img_bbox_pairs, output_h5_file, feature_layer='fc7',
                                  copy_from_file=None):
    """
    image_root = dataset.image_root
    image_infos = {img_id_1: {}, img_id_2: {}}
    img_bbox_pairs = [(img_id, [x1,y1,x2,y2]), ...]
    """
    if os.path.exists(output_h5_file):
      print ("Output file already exists: %s" % output_h5_file)
      h5file = h5py.File(output_h5_file, 'r+')
      extracted_img_bbox_pairs = set([i for i in h5file.keys() if i != 'imgs_with_errors'])
    else:
      dirname = os.path.dirname(output_h5_file)
      if not os.path.exists(dirname):
        os.makedirs(dirname)
      h5file = h5py.File(output_h5_file, 'w')
      extracted_img_bbox_pairs = set([])

    if copy_from_file:
      h5_file_to_copy_from = h5py.File(copy_from_file,'r')
      num_copied_items = 0
      for pair in img_bbox_pairs:
        if not (str(pair) in h5file) and str(pair) in h5_file_to_copy_from:
          h5dataset = h5file.create_dataset(str(pair), shape=h5_file_to_copy_from[str(pair)].shape,
                                            dtype=h5_file_to_copy_from[str(pair)].dtype)
          h5dataset[:] = h5_file_to_copy_from[str(pair)][:]
          extracted_img_bbox_pairs.add(str(pair))
          num_copied_items += 1
      print "Copied %d/%d items from another file" % (num_copied_items, len(img_bbox_pairs))

    remaining_img_bbox_pairs = []
    for pair in img_bbox_pairs:
      if str(pair) not in extracted_img_bbox_pairs:
        remaining_img_bbox_pairs.append(pair)
    img_bbox_pairs = remaining_img_bbox_pairs

    num_pairs = len(img_bbox_pairs)
    if num_pairs>0:
      if not self.initialized: self.init()
      batch_indices = range(0,num_pairs,self.BATCH_SIZE) + [num_pairs]
      features = np.zeros((1,self.net.blobs[feature_layer].data.shape[1]))
      curr_img_id = None
      if 'imgs_with_errors' in h5file:
        imgs_with_errors = h5file['imgs_with_errors'][:]
        del h5file['imgs_with_errors']
      else:
        imgs_with_errors = np.array([])

      for i in range(len(batch_indices)-1):
        batch_start_id = batch_indices[i]
        batch_end_id = batch_indices[i+1]
        sys.stdout.write("\rProcessing batch %d of %d" % (i+1, len(batch_indices)-1))
        sys.stdout.flush()
        for j in range(batch_start_id,batch_end_id):
          img_error = False
          if curr_img_id is None or curr_img_id != img_bbox_pairs[j][0]:
            curr_img_id = img_bbox_pairs[j][0]
            img_filename = '%s/%s' % (image_root, image_infos[curr_img_id]['file_name'])
            try:
              img = caffe.io.load_image(img_filename)
            except:
              print "Problem loading %s" % curr_img_id
              imgs_with_errors = np.hstack((imgs_with_errors,curr_img_id))
              img_error = True
          if img_error: continue
          bbox = img_bbox_pairs[j][1]
          img_ht = img.shape[0]
          img_wd = img.shape[1]
          roi = [max(bbox[0]-4,0), max(bbox[1]-4,0), min(bbox[0]+bbox[2]+4,img_wd-1), min(bbox[1]+bbox[3]+4,img_ht-1)]
          roi = [int(pt) for pt in roi]
          cropped_img = img[roi[1]:roi[3], roi[0]:roi[2],:]
          num_rows = cropped_img.shape[0]
          num_cols = cropped_img.shape[1]
          if num_rows > num_cols:
            padded_img = np.tile([123.68/255, 116.779/255, 103.939/255],(num_rows,num_rows,1))
            padded_img[:,(num_rows-num_cols)/2:(num_rows-num_cols)/2+num_cols,:] = cropped_img
          else:
            padded_img = np.tile([123.68/255, 116.779/255, 103.939/255],(num_cols,num_cols,1))
            padded_img[(num_cols-num_rows)/2:(num_cols-num_rows)/2+num_rows,:,:] = cropped_img
          self.net.blobs['data'].data[j-batch_start_id][:] = self.transformer.preprocess('data', padded_img)
        self.net.forward()
        for j in range(batch_start_id,batch_end_id):
          if str(img_bbox_pairs[j]) not in h5file:
            h5dataset = h5file.create_dataset(str(img_bbox_pairs[j]), shape=(1,features.shape[1]), dtype=features.dtype)
            h5dataset[:] = self.net.blobs[feature_layer].data[j-batch_start_id,:]
          else:
            print "Duplicate %s found" % str(img_bbox_pairs[j])

      sys.stdout.write('\n')
      errors = h5file.create_dataset('imgs_with_errors',shape=imgs_with_errors.shape,dtype=imgs_with_errors.dtype)
      errors[:] = imgs_with_errors
    h5file.close()

image_feature_extractor = ImageFeatureExtractor(image_net='vggnet')
