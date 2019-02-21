from data_batcher import SantanderDataObject
import tensorflow as tf
import numpy as np

class SantanderVanillaModel(object):
    def __init__(self,FLAGS):
        self.FLAGS=FLAGS
        self.dataObject=SantanderDataObject(self.FLAGS.batch_size,self.FLAGS.test_size)

        with tf.variable_scope("SantanderModel",initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,uniform=True)):
            self.add_placeholders()
            self.build_graph()
            self.add_loss()
            self.add_training_step()

    def add_placeholders(self):
        self.x=tf.placeholder(dtype=tf.float32,shape=[None,200])
        self.y=tf.placeholder(dtype=tf.float32,shape=[None,1])
        self.keep_prob=tf.placeholder_with_default(1.0, shape=())

    def build_graph(self):
        HIDDEN_LAYER_1=4096
        HIDDEN_LAYER_2=4096
        HIDDEN_LAYER_3=4096

        output1=tf.contrib.layers.fully_connected(self.x,HIDDEN_LAYER_1,activation_fn=tf.nn.relu)
        output2=tf.contrib.layers.fully_connected(output1,HIDDEN_LAYER_2,activation_fn=tf.nn.relu)
        output3=tf.contrib.layers.fully_connected(output2,HIDDEN_LAYER_3,activation_fn=tf.nn.relu)
        self.final_output=tf.contrib.layers.fully_connected(output3,1,activation_fn=None)
        self.logits=tf.identity(self.final_output,name='logits')

    def add_loss(self):
        self.loss=tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y,logits=self.logits)
        self.cost=tf.reduce_mean(self.loss)
        self.prediction=tf.nn.sigmoid(self.final_output)
        self.correct_pred=tf.equal(tf.round(self.prediction),self.y)
        self.accuracy=tf.reduce_mean(tf.cast(self.correct_pred,tf.float32))

    def add_training_step(self):
        self.train_step=tf.train.AdamOptimizer(learning_rate=self.FLAGS.learning_rate).minimize(self.cost)

    def run_train_iter(self,sess,x,y):
        train_data_feed={
            self.x:x,
            self.y:y,
            self.keep_prob:(1.0-self.FLAGS.dropout),
        }
        sess.run(self.train_step,feed_dict=train_data_feed)

    def get_validation_accuracy(self,sess):
        validation_accuracy=0.0
        total_items=0
        for x,y in self.dataObject.generate_dev_data():
            total_items+=x.shape[0]
            dev_data_feed={
                self.x:x,
                self.y:y,
                self.keep_prob:1.0,
            }
            validation_accuracy_batch=sess.run([self.accuracy],dev_data_feed)
            validation_accuracy += validation_accuracy_batch[0]*x.shape[0]
        validation_accuracy/=total_items
        return validation_accuracy

    def get_validation_predictions(self,sess):
        output=[]
        values=[]
        for x,y in self.dataObject.generate_dev_data():
            dev_data_feed={
                self.x:x,
                self.keep_prob:1.0,
            }
            dev_output=sess.run(self.prediction,feed_dict=dev_data_feed)
            dev_output=np.squeeze(dev_output )
            output.extend(dev_output.tolist())
            values.extend(y)
        return output,values

    def get_test_data(self,sess):
        output=[]
        for x in self.dataObject.generate_test_data():
            test_data_feed={
                self.x:x,
                self.keep_prob:1.0,
            }
            test_output=sess.run(self.prediction,feed_dict=test_data_feed)
            test_output=np.squeeze(test_output)
            output.extend(test_output.tolist())
        return self.dataObject.test_ids.tolist(),output

    def run_epoch(self,sess):
        for x,y in self.dataObject.generate_one_epoch():
            self.run_train_iter(sess,x,y)
        validation_accuracy=self.get_validation_accuracy(sess)
        return validation_accuracy
