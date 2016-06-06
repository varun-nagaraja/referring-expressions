# The original version of this file was written by Jeff Donahue (jdonahue@eecs.berkeley.edu)

import h5py
import numpy as np
import os
import sys
import distutils.util

class SequenceGenerator:
  def __init__(self):
    self.dimension = 10
    self.batch_stream_length = 2000
    self.batch_num_streams = 8
    self.min_stream_length = 13
    self.max_stream_length = 17
    self.substream_names = None
    self.streams_initialized = False
    self.num_completed_streams = 0

  def streams_exhausted(self):
    return False

  def init_streams(self):
    self.streams = [None] * self.batch_num_streams
    self.stream_indices = [0] * self.batch_num_streams
    self.reset_stream(0)
    self.streams_initialized = True

  def reset_stream(self, stream_index):
    streams = self.get_streams()
    stream_names = sorted(streams.keys())
    if self.substream_names is None:
      assert len(stream_names) > 0
      self.substream_names = stream_names
    assert self.substream_names == stream_names
    if self.streams[stream_index] is None:
      self.streams[stream_index] = {}
    for k, v in streams.iteritems():
      self.streams[stream_index][k] = v
    self.stream_indices[stream_index] = 0

  # Pad with zeroes by default -- override this to pad with something else
  # for a particular stream
  def get_pad_value(self, stream_name):
    return -1

  def swap_axis(self, stream_name):
    return False

  def get_next_batch(self, truncate_at_exhaustion=True):
    if not self.streams_initialized:
      self.init_streams()
    batch = {}
    reached_exhaustion = False

    # Used when writing multiple files
    for i in range(self.batch_num_streams):
      # The first stream is already loaded by reset_stream
      if self.num_completed_streams > 0:
        self.reset_stream(i)
      for name in self.substream_names:
        if name in batch:
          batch[name] = np.vstack((batch[name], np.array([self.streams[i][name]])))
        else:
          batch[name] = np.array([self.streams[i][name]])
      self.num_completed_streams += 1
      reached_exhaustion = reached_exhaustion or self.streams_exhausted()
      if reached_exhaustion and truncate_at_exhaustion:
        break

    for key, batch_data in batch.iteritems():
      if self.swap_axis(key):
        batch[key] = np.swapaxes(batch_data, 0, 1)

    if reached_exhaustion:
      print ('Exhausted all data; cutting off at %d batches with %d streams completed' %
             (np.ceil(self.num_completed_streams / float(self.batch_num_streams)), self.num_completed_streams))

    return batch

  def get_streams(self):
    raise Exception('get_streams should be overridden to return a dict ' +
                    'of equal-length iterables.')


class HDF5SequenceWriter:
  def __init__(self, sequence_generator, output_dir=None, verbose=False):
    self.generator = sequence_generator
    assert output_dir is not None  # required
    self.output_dir = output_dir
    if os.path.exists(output_dir + '/hdf5_chunk_list.txt'):
      print('Output directory already exists: ' + output_dir)
      while True:
        sys.stdout.write('Do you want to overwrite it? [Y/n] ')
        choice = raw_input().lower()
        try:
          if distutils.util.strtobool(choice):
            break
          else:
            print "Exiting"
            sys.quit(0)
        except ValueError:
          pass
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    self.verbose = verbose
    self.filenames = []
    self.batch_index = 0

  def write_batch(self, stop_at_exhaustion=True):
    batch = self.generator.get_next_batch(truncate_at_exhaustion=stop_at_exhaustion)
    filename = '%s/batch_%d.h5' % (self.output_dir, self.batch_index)
    self.filenames.append(filename)
    h5file = h5py.File(filename, 'w')
    dataset = h5file.create_dataset('buffer_size', shape=(1,), dtype=np.int)
    dataset[:] = self.generator.batch_num_streams
    for key, batch_data in batch.iteritems():
      if self.verbose:
        for s in range(self.generator.batch_num_streams):
          stream = np.array(self.generator.streams[s][key])
          print 'batch %d, stream %s, index %d: ' % (self.batch_index, key, s), stream
      h5dataset = h5file.create_dataset(key, shape=batch_data.shape, dtype=batch_data.dtype)
      h5dataset[:] = batch_data
    h5file.close()
    self.batch_index += 1

  def write_to_exhaustion(self):
    while not self.generator.streams_exhausted():
      self.write_batch(stop_at_exhaustion=True)

  def write_filelists(self):
    assert self.filenames is not None
    filelist_filename = '%s/hdf5_chunk_list.txt' % self.output_dir
    with open(filelist_filename, 'w') as listfile:
      for filename in self.filenames:
        listfile.write('%s\n' % filename)
