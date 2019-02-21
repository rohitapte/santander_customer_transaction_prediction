import tensorflow as tf
import os
from neural_models import SantanderVanillaModel
import pandas as pd

tf.app.flags.DEFINE_integer("gpu", 1, "Which GPU to use, if you have multiple.")
tf.app.flags.DEFINE_integer("num_epochs",20, "Number of epochs to train. 0 means train indefinitely")
tf.app.flags.DEFINE_float("learning_rate",0.001,"Learning rate.")
tf.app.flags.DEFINE_float("dropout",0.5,"Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size",10000,"Batch size to use")
tf.app.flags.DEFINE_float("test_size",0.10,"Dev set to split from training set")

FLAGS = tf.app.flags.FLAGS
os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)
config=tf.ConfigProto()
config.gpu_options.allow_growth = True

santander_model=SantanderVanillaModel(FLAGS)
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)
for epoch in range(FLAGS.num_epochs):
    validation_accuracy = santander_model.run_epoch(sess)
    print('validation_accuracy for epoch ' + str(epoch) + ' => ' + str(validation_accuracy))

print('Final validation_accuracy => ' +str(santander_model.get_validation_accuracy(sess)))
test_ids,test_probs=santander_model.get_test_data(sess)
df=pd.DataFrame(
    {'ID_code':test_ids,
     'target':test_probs}
)
df.to_csv('neural_submission.csv',index=None)