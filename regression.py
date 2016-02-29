import tensorflow as tf
import mahotas
import skimage.io as imgio
import numpy
import scipy.misc
tf.set_random_seed(42)

from models import RegressionProblem

class RegressionProblem():
    def __init__(self,in_dim,out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.batchsize = 32
        self.stdev = 0.25
        self.input_placeholder = tf.placeholder('float',[self.batchsize,self.in_dim],name='input',)

        self.preprocessed_data = self.input_placeholder
    
        self.learningRate = 0.01
        self.true_params = tf.Variable(numpy.random.randn(self.out_dim,self.in_dim).astype('float32'),name='true_params',trainable=False)
        self.params = tf.Variable(tf.zeros([self.out_dim,self.in_dim]),name = 'regression_weights') 
       

    def gen_example(self): 
        return numpy.random.randn(self.batchsize,self.in_dim)

    def predict(self,data,weights):
        return tf.matmul(data,weights,False,True)#,True,False

    def generate(self,data,weights):
        predicted = self.predict(data,weights)
        noise = tf.random_normal(predicted.get_shape(),0,self.stdev)
        return predicted + noise


def main():
    tf.set_random_seed(32)
    in_dim = 5
    out_dim = 1
    problem = RegressionProblem(out_dim,in_dim)
    input_placeholder = problem.input_placeholder

    truth = problem.generate(problem.preprocessed_data,problem.true_params)
    prediction = problem.predict(problem.preprocessed_data,problem.params)
    op_loss = tf.reduce_mean(tf.square(prediction - truth))
    loss_summary = tf.scalar_summary('loss',op_loss)

    optimizer = tf.train.GradientDescentOptimizer(problem.learningRate)

    tvars = tf.trainable_variables()
    grads = tf.gradients(op_loss, tvars)
    grads = [tf.clip_by_norm(grad,10.0) for grad in grads]
    op_optimize = optimizer.apply_gradients(zip(grads, tvars))
    
    sw = tf.train.SummaryWriter('./logs/')

    summaries = tf.merge_all_summaries()

    numIters = 1000
    op_init = tf.initialize_all_variables()

    with tf.Session() as session:
        session.run(op_init)
        for iteration in range(numIters):
            example = problem.gen_example()
            error,summary, _  = session.run([op_loss, summaries, op_optimize], feed_dict={input_placeholder: example})
            print("%s: %s" % (iteration, error))
            sw.add_summary(summary)

    sw.flush()
    sw.close()
        
if __name__ == '__main__':
    main()
