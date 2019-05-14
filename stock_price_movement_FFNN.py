# Project Title: Stock Price Movement Prediction Using FFNN
# Project Type: Prediction of stock price movement

# ---------------------- libraries -------------------------------
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn import metrics

# -------------------------- functions ---------------------------

# function to reset graph for each run
def reset_graph(seed=42):
  tf.reset_default_graph()
  tf.set_random_seed(seed)
  np.random.seed(seed)

reset_graph()

# define a function for reshaping
# input is the df to be reshaped
# window_size set to default value of 50
def reshape(df,window_size=50):
    df_as_array=np.array(df)
    temp = np.array([np.arange(i-window_size,i) for i in range(window_size,df.shape[0])])
    new_df = df_as_array[temp[0:len(temp)]]
    new_df2 = new_df.reshape(len(temp),8*window_size)
    return new_df2

def get_model_params():
    gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    return {gvar.op.name: value for gvar, value in zip(gvars, tf.get_default_session().run(gvars))}

def restore_model_params(model_params):
    gvar_names = list(model_params.keys())
    assign_ops = {gvar_name: tf.get_default_graph().get_operation_by_name(gvar_name + "/Assign")
                  for gvar_name in gvar_names}
    init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
    feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
    tf.get_default_session().run(assign_ops, feed_dict=feed_dict)

# --------------------- read and process data -----------------------

# read in data
# data was processed in the Jyputer Notebook file and then saved as csv files
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

'''
reshaped data, so that consistent with CNN
'''
# get inputs for x and y
col_for_x = ['mix_mv_avg','5_price_diff','mv_avg_diff', 'avg_quantity','quantity_price','ct_rising','aux_flag','price']

x_train = train[col_for_x]


x_train_all = reshape(x_train)
y_train = train.label[:-50]
y_train_all = np.array(pd.get_dummies(y_train)) # transform the label into one hot encoding


# set up test set
x_test = test[col_for_x]
x_test = reshape(x_test)
y_test = test.label[:-50]


# --------------------- configurations ----------------------------------


n_inputs = x_train_all.shape[1]
n_hidden1 = 300  # number of neurons for each hidden layer
n_hidden2 = 100
#n_hidden3 = 100
#n_hidden4 = 50
n_outputs = 3

learning_rate = 0.0003
n_epochs = 200
batch_size = 64

# parameters for early stopping
iteration = 0
best_loss_val = np.infty
check_interval = 10
checks_since_last_progress = 0
max_checks_without_progress = 20 #20
best_model_params = None

# --------------------------- FFNN -------------------------------------

# place holders for X and y
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None,n_outputs), name="y")

# place holder for keep probability (for drop out regularization)
#keep_prob = tf.placeholder(tf.float32)


with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1",
                              activation=tf.nn.relu) # use relu as activation function

    #hidden1_drop = tf.nn.dropout(hidden1, keep_prob)

    hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2",
                             activation=tf.nn.relu) # use output from hidden1 as input

    #hidden2_drop = tf.nn.dropout(hidden2, keep_prob)

    #hidden3 = tf.layers.dense(hidden2, n_hidden3, name="hidden3",
                             #activation=tf.nn.relu)

    #hidden3_drop = tf.nn.dropout(hidden3, keep_prob)

    #hidden4 = tf.layers.dense(hidden3, n_hidden4, name="hidden4",
                              #activation=tf.nn.relu)

    logits = tf.layers.dense(hidden2, n_outputs, name="outputs")


with tf.name_scope("loss"):
    # use cross entropy as loss function
    xentropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss") # mean cross entropy over all instances
    loss_summary = tf.summary.scalar('log_loss', loss)


with tf.name_scope("train"):
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate) # define optimizer
    #optimizer = tf.train.AdagradOptimizer(learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    #optimizer = tf.train.RMSPropOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)


with tf.name_scope("eval"):

    prob = tf.nn.softmax(logits, name = 'probability')
    # check if the position of the highest probability == the position of the correct number in one hot encoding
    # if the same --> right prediction, return True
    correctPrediction = tf.equal(tf.argmax(prob, 1),tf.argmax(y, 1))
    # accuracy is the mean of the boolean values
    accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))
    #correct = tf.nn.in_top_k(logits, y, 1) # return a 1-d tensor (vector/array) with boolean values
    accuracy_summary = tf.summary.scalar('accuracy', accuracy)


init = tf.global_variables_initializer()
saver = tf.train.Saver()

# -------------------------- train model ----------------------------

# split entire train data into train and validation sets sequentially at a ratio of 8:2
x_train_input = x_train_all[0:10935,:]
x_val_input = x_train_all[10935:,:]
y_train_input = y_train_all[0:10935,:]
y_val_input = y_train_all[10935:,:]


# The following commented out codes are for graphs and scalars to Tensorboard when tuning hyper parameters
'''
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('./graphs_ffl/train/op_rms',tf.get_default_graph())
val_writer = tf.summary.FileWriter('./graphs_ffl/val/op_rms',tf.get_default_graph())
'''
# a writer to write the FFN graph Tensorboard
#writer = tf.summary.FileWriter('./graphs_ffl', tf.get_default_graph())

start = time.time()

with tf.Session() as sess:  # train model (train set and validation set)
    init.run()
    for epoch in range(n_epochs):

        for i in range(0, len(x_train_input), batch_size):
            X_batch = x_train_input[i: i+batch_size]
            y_batch = y_train_input[i: i+batch_size]
            #sess.run(training_op, feed_dict = {X:X_batch, y:y_batch,keep_prob:0.3})
            sess.run(training_op, feed_dict = {X:X_batch, y:y_batch})

        if iteration % check_interval == 0:
                loss_val = loss.eval(feed_dict={X: x_val_input, y: y_val_input})
                if loss_val < best_loss_val:
                    best_loss_val = loss_val
                    checks_since_last_progress = 0
                    best_model_params = get_model_params()
                else:
                    checks_since_last_progress += 1

        # train accuracy and loss
        train_loss = sess.run(loss, feed_dict = {X:X_batch, y:y_batch})
        #train_summary,train_loss = sess.run([merged,loss], feed_dict = {X:X_batch, y:y_batch})
        acc_train = accuracy.eval(feed_dict = {X:X_batch, y:y_batch})
        #train_writer.add_summary(train_summary, epoch)

        # validation set loss and accuracy
        #val_summary,val_loss = sess.run([merged,loss], feed_dict = {X:x_val_input, y:y_val_input})
        val_loss = sess.run(loss, feed_dict = {X:x_val_input, y:y_val_input})
        acc_val = accuracy.eval(feed_dict = {X:x_val_input, y:y_val_input})
        #val_writer.add_summary(val_summary, epoch)

        # print result for each epoch
        print("epoch {}: ".format(epoch+1),
              "Train loss: {}".format(train_loss),
              "Train accuracy: {}".format(acc_train),
              "Validation loss: {}".format(val_loss),
              "Validation accuracy: {}".format(acc_val))

        # break loop if no improvement after number of max_checks
        if checks_since_last_progress > max_checks_without_progress:
            print("Early stopping!")
            break

    # restore the parameters of the best model (the one with the lowest loss)
    if best_model_params:
        restore_model_params(best_model_params)

    acc_val_final = accuracy.eval(feed_dict={X: x_val_input, y: y_val_input})
    print("Final accuracy on validation set:", acc_val_final)

    save_path = saver.save(sess, "./ffn_model3.ckpt")

#writer.close()
#train_writer.close()
#val_writer.close()

'''

# train model on entire train data
# commented out the following code so that can run separately after restarting kernel

with tf.Session() as sess:  # train model
    init.run()
    for epoch in range(n_epochs):

        for i in range(0, len(x_train_all), batch_size):
            X_batch = x_train_all[i: i+batch_size]
            y_batch = y_train_all[i: i+batch_size]
            _, train_loss = sess.run([training_op,loss], feed_dict = {X:X_batch, y:y_batch})
        acc_train = accuracy.eval(feed_dict = {X:X_batch, y:y_batch})

        #print(epoch, "Train accuracy:", acc_train)
        print("epoch {}: ".format(epoch+1),
              "Train loss: {}".format(train_loss),
              "Train accuracy: {}".format(acc_train)
              )

    save_path = saver.save(sess, "./ffn_model.ckpt")


'''

# -------------------------- make predictions ----------------------------

# use model to predict on test set
with tf.Session() as sess:
    saver.restore(sess, "./ffn_model3.ckpt") # load model parameters from disk
    Z = prob.eval(feed_dict = {X: x_test})
    y_pred = np.argmax(Z, axis = 1) - 1

# print results
print(pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted']))
print(metrics.classification_report(y_test,y_pred))
print("Accuracy: " + str(metrics.accuracy_score(y_test,y_pred)))
print("Cohen's Kappa: " + str(metrics.cohen_kappa_score(y_test,y_pred)))
print('Took: %f seconds' %(time.time() - start))
