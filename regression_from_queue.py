import tensorflow as tf
import numpy
import scipy.misc
tf.set_random_seed(42)
from util import *
from models import RegressionProblem
from sys import exit

def main():
  batch_size = 32
  min_after_dequeue = 0
  capacity = min_after_dequeue + 3 * batch_size
  learningRate = 0.1

  label_dimension = 3
  features_dimension = 5
  num_epochs = 15
  batches_per_epoch = 5
  num_test_set_batches = 25

  label_files = ['data/file0.labels.csv']
  features_files = ['data/file0.feats.csv']
  train_feats, train_labels = gen_batcher(features_files,features_dimension,label_files,label_dimension,batch_size=batch_size, shuffle=True)

  test_label_files = ['data/file1.labels.csv']
  test_features_files = ['data/file1.feats.csv']
  test_feats, test_labels = gen_batcher(test_features_files,features_dimension,test_label_files,label_dimension,batch_size=batch_size,shuffle=False)

  def make_model(input_features):
    weights =  tf.get_variable("weights", [label_dimension,features_dimension],initializer=tf.random_normal_initializer())
    predict = tf.matmul(input_features,weights,False,True,name='mul')
    return predict

  def make_model_clone(input_features):
    with tf.variable_scope("regression_model") as scope:
      scope.reuse_variables()
      return make_model(input_features)

  with tf.variable_scope("regression_model") as scope:
     prediction_for_training = make_model(train_feats)
  
  def make_error(prediction, labels):
    return tf.reduce_mean(tf.square(prediction - labels))

  op_loss   = make_error(prediction_for_training, train_labels)

  optimizer = tf.train.GradientDescentOptimizer(learningRate)
  tvars = tf.trainable_variables()
  grads = tf.gradients(op_loss, tvars)
  op_optimize = optimizer.apply_gradients(zip(grads, tvars))
  
  #note that this assumes that the number of test examples is divisible by minibatch size
  test_labels_onepass, reset_labels  = single_pass(test_labels,num_test_set_batches)
  test_feats_onepass, reset_features = single_pass(test_feats ,num_test_set_batches)

  pred_test = make_model_clone(test_feats_onepass)

  eval_loss = make_error(pred_test,test_labels_onepass)

  def evaluate(session):
    count = 0
    total = 0.
    session.run([reset_labels,reset_features])
    while(True):
      try:
        e = session.run(eval_loss)
        count = count + 1
        total =  total + e
      except tf.errors.OutOfRangeError:
        break
    return total/count


  op_init = tf.initialize_all_variables()
  with tf.Session() as session:
    session.run(op_init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=session,coord=coord)

    for epoch in range(num_epochs):
      for step in range(batches_per_epoch):
        loss,_ = session.run([op_loss,op_optimize])

      test_error = evaluate(session)
      print('test error = %f' % test_error)


    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=0.05)   

if __name__ == '__main__':
    main()
