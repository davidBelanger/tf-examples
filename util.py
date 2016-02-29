import tensorflow as tf
import numpy as np
#todo: extend this to operate on multiple files

#this is an iterator that takes a single pass over the data. If the batch_size doesn't divide the number of examples, then the final minibatch is smaller than batch_size
class MinibatchIterator():
  def __init__(self,data_array,batch_size):
    self.data_array = data_array
    self.num_rows = data_array.shape[0]
    self.batch_size = batch_size
    self.start_idx = 0

  def __iter__(self):
    self.start_idx = 0
    return self


  def next(self): 
    if self.start_idx >= self.num_rows:
      raise StopIteration
    else:
        end_idx = min(self.start_idx + self.batch_size, self.num_rows)
        to_return = self.data_array[self.start_idx:end_idx] #TODO: should this be inclusive?
        self.start_idx = end_idx
        return to_return


#this assumes that the loss is an average per-batch loss, eg from reduce_mean (rather than reduce_sum)
def evaluate_from_iterator(session,eval_op,feature_batcher,label_batcher,test_feat_placeholder,test_label_placeholder,feed_dict={}):
  count = 0
  total = 0.
  for (feats,labels) in zip(feature_batcher,label_batcher):
    num = feats.shape[0]
    count = count + num
    fd = merge_two_dicts(feed_dict,{test_feat_placeholder: feats,test_label_placeholder:labels})
    e = session.run(eval_op,feed_dict=fd)
    total = total + e*num
  return total/count

def single_pass(source_batcher,num_batches):
    zero = tf.constant(0, dtype=tf.int64)
    batch_count = tf.Variable(zero, name="epochs", trainable=False)
    limiter = tf.count_up_to(batch_count,num_batches)
    with tf.control_dependencies([limiter]):
      batcher = tf.identity(source_batcher)

    reset = tf.assign(batch_count, zero)

    return batcher, reset

def data_from_csv(file_names,dimension):
  filename_queue = tf.train.string_input_producer(file_names)
  
  reader = tf.TextLineReader()
  key, value = reader.read(filename_queue)
  defaults = [[0.0]]*dimension #todo: this assumes the data is floats
  columns = tf.decode_csv(value,record_defaults=defaults)
  return tf.pack(columns)


def gen_batcher(features_files,features_dimension,label_files,label_dimension,batch_size,shuffle):
  feats_loader = data_from_csv(features_files,features_dimension)
  labels_loader = data_from_csv(label_files,label_dimension)
  
  min_after_dequeue = 100
  capacity = min_after_dequeue + 3 * batch_size

  #this returns feats, labels
  if(shuffle):
    return tf.train.shuffle_batch([feats_loader,labels_loader],batch_size=batch_size,capacity=capacity,min_after_dequeue=min_after_dequeue)
  else:
    return tf.train.batch([feats_loader,labels_loader],batch_size=batch_size,capacity=capacity)


#taken from http://stackoverflow.com/questions/38987/how-can-i-merge-two-python-dictionaries-in-a-single-expression
def merge_two_dicts(x, y):
    '''Given two dicts, merge them into a new dict as a shallow copy.'''
    z = x.copy()
    z.update(y)
    return z
