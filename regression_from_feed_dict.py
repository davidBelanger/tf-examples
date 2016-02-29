import tensorflow as tf
import numpy
import scipy.misc
tf.set_random_seed(42)
from util import *

def main():
  batch_size = 32
  min_after_dequeue = 0
  capacity = min_after_dequeue + 3 * batch_size

  label_dimension = 3
  features_dimension = 5
  num_epochs = 100
  batches_per_epoch = 2
  test_batch_size = 32

  label_files = ['file0.labels.csv']
  features_files = ['file0.feats.csv']
  train_feats, train_labels = gen_batcher(features_files,features_dimension,label_files,label_dimension,batch_size=batch_size, shuffle=True)

  #change this into a python loader?
  test_features_data = np.loadtxt('file1.feats.csv',delimiter=',')
  test_labels_data   = np.loadtxt('file1.labels.csv',delimiter=',')
  test_feats  = MinibatchIterator(test_features_data,test_batch_size)
  test_labels = MinibatchIterator(test_labels_data,test_batch_size)


  test_label_placeholder = tf.placeholder('float32',[None,label_dimension])
  test_feat_placeholder = tf.placeholder('float32',[None,features_dimension])


  def make_model(input_features):
    weights =  tf.get_variable("weights", [label_dimension,features_dimension],initializer=tf.random_normal_initializer())
    predict = tf.matmul(input_features,weights,False,True,name='mul')
    return predict

  with tf.variable_scope("regression_model") as scope:
    prediction_for_training = make_model(train_feats)
    scope.reuse_variables()
    prediction_for_testing = make_model(test_feat_placeholder)

  op_loss = tf.reduce_mean(tf.square(prediction_for_training - train_labels))

  learningRate = 0.1
  optimizer = tf.train.GradientDescentOptimizer(learningRate)

  tvars = tf.trainable_variables()
  grads = tf.gradients(op_loss, tvars)
  op_optimize = optimizer.apply_gradients(zip(grads, tvars))
  
  eval_loss = tf.reduce_mean(tf.square(prediction_for_testing - test_label_placeholder))

  def evaluate_mse(session):
   return evaluate_from_iterator(session,eval_loss,test_feats,test_labels,test_feat_placeholder,test_label_placeholder)


  op_init = tf.initialize_all_variables()
  count = 0
  with tf.Session() as session:
    session.run(op_init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=session,coord=coord)

    for epoch in range(num_epochs):
      for step in range(batches_per_epoch):
        loss,_ = session.run([op_loss,op_optimize])

      test_error = evaluate_mse(session)
      print('test error = %f' % test_error)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=0.05)   

if __name__ == '__main__':
    main()
