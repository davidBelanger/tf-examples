# tf-examples

This is a set of simple applications in tensorflow designed to demonstrate how to deal with data loading, logging etc. Check out:

* regression.py: simple linear regression problem that generates training examples as part of the computation graph
* logistic_regression.py: simple logistic regression problem that also generates training examples on the fly as part of the graph. 
* regression_from_feed_dict.py linear regression where the data is loaded from csv files. Creating minibatches of training data is managed as part of the computation graph. At test time, we process the data sequentially using a feed_dict. 
* regression_from_queue.py: this asynchrounously processes the train and test data both using tensorflow queues. It only works if the number of test examples is divisible by the number of examples in your test set. 
* util.py: useful utility code
