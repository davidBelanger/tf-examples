import tensorflow as tf
import mahotas
import skimage.io as imgio
import numpy
import scipy.misc
tf.set_random_seed(42)


class regression_problem():
    def __init__(self):
        self.dimension = 5
        self.batchsize = 32
        
        self.input_placeholder = tf.placeholder('float',[self.batchsize,self.dimension],name='input',)

        self.preprocessed_data = self.input_placeholder
    
        self.learningRate = 0.1
        self.true_params = tf.constant(numpy.random.randn(1,self.dimension).astype('float32'),name='true_params')
        self.params = tf.Variable(tf.zeros([1,self.dimension]),name = 'regression_weights') 
       

    def gen_example(self): 
        return numpy.random.randn(self.batchsize,self.dimension)

    def predict_logit(self,data,weights):
        return tf.matmul(data,weights,False,True)

    def predict_probability(self,data,weights):
        return tf.sigmoid(self.predict_logit(data,weights))

    #this draws data from the conditional probability model
    def generate(self,data,weights):
        stdev = 0.1
        predicted = self.predict_probability(data,weights)
        unif = tf.random_uniform(predicted.get_shape(), minval=0, maxval=1, dtype=tf.float32)
        bits = tf.less(unif,predicted)
        return tf.to_float(bits)



def main():
    tf.set_random_seed(32)

    problem = regression_problem()
    input_placeholder = problem.input_placeholder

    truth = problem.generate(problem.preprocessed_data,problem.true_params)
    prediction = problem.predict_logit(problem.preprocessed_data,problem.params)
    op_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(prediction,truth))
    loss_summary = tf.scalar_summary('loss',op_loss)

    optimizer = tf.train.GradientDescentOptimizer(problem.learningRate)

    tvars = tf.trainable_variables()
    grads = tf.gradients(op_loss, tvars)
    op_optimize = optimizer.apply_gradients(zip(grads, tvars))
    
    parameter_loss = tf.reduce_mean(tf.square(problem.true_params - problem.params))
    sw = tf.train.SummaryWriter('./logs/')

    summaries = tf.merge_all_summaries()

    numIters = 1000
    op_init = tf.initialize_all_variables()

    with tf.Session() as session:
        session.run(op_init)
        for iteration in range(numIters):
            example = problem.gen_example()
            error,summary, sql, _  = session.run([op_loss, summaries, parameter_loss,op_optimize], feed_dict={input_placeholder: example})
            print("%s: %f %f" % (iteration, error,sql))
            sw.add_summary(summary)

    sw.flush()
    sw.close()
        
if __name__ == '__main__':
    main()
