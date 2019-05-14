# Project Title: Stock Price Movement Prediction Using RNN
# Project Type: Prediction of stock price movement


# ------------------------ Libaries used --------------------------------------
import tensorflow as tf
import pandas as pd
import os as os
import numpy as np
#from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt
import time
from sklearn import metrics
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# ------------------------ functions ------------------------------------------

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

def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

def batch_func(X, y, batch_size):
    batches = []
    for i in range(0, len(X), batch_size):
        X_batch = X[i: i+batch_size]
        y_batch = y[i: i+batch_size]
        mini_batch = (X_batch, y_batch)
        batches.append(mini_batch)
    return batches

# -------------------- configs ------------------------------------------------

n_steps = 50     # 50 time steps, each step corresponds to 1*8 (a row) of a image
n_inputs = 8    # the size of the input vector
n_neurons = 150  # recurrent neurons/The number of units in the RNN cell #150
n_outputs = 3   # number of neurons/units of the fully connected layer
n_layers = 3# use for additional layers

learning_rate = 0.001
batch_size = 50 #tried 128, worse than 50
n_epochs = 10

train_keep_prob = 0.5

iteration = 0


best_loss_val = np.infty
check_interval = 10
checks_since_last_progress = 0
max_checks_without_progress = 10 #20
best_model_params = None

# ----------------- read in data & reshape data  ------------------------------
# read in data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


# get inputs for x and y
col_for_x = ['mix_mv_avg','5_price_diff','mv_avg_diff', 'avg_quantity','quantity_price','ct_rising','aux_flag','price']

x_train = train[col_for_x]

#x_train_all = x_train.values[:,:]
x_train_all = reshape(x_train)
x_train_all = x_train_all.reshape((x_train_all.shape[0],n_steps,n_inputs)) # reshape for RNN
y_train = train.label[:-50]
y_train_all = np.array(pd.get_dummies(y_train)) # transform the label into one hot encoding


# set up test set
x_test = test[col_for_x]
x_test = reshape(x_test)
x_test = x_test.reshape((x_test.shape[0],n_steps,n_inputs))
y_test = test.label[:-50]

# split entire train data into train and validation sets sequentially at a ratio of 8:2
x_train_input = x_train_all[0:10935,:]
x_val_input = x_train_all[10935:,:]
y_train_input = y_train_all[0:10935,:]
y_val_input = y_train_all[10935:,:]


# -------------------------- RNN ----------------------------------------------

tf.reset_default_graph()

keep_prob = tf.placeholder_with_default(1.0, shape=())

with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs], name="X")
    y = tf.placeholder(tf.int32, [None, n_outputs], name="y")
#    training = tf.placeholder_with_default(False, shape=[], name='training')

with tf.name_scope("RNN"):

    # try LSTM

#   lstm_cells = [tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons, activation=tf.nn.relu)
#                for layer in range(n_layers)]
#   lstm_cells_drop = [tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_prob) for cell in lstm_cells]
#   multi_layer_cell = tf.contrib.rnn.MultiRNNCell(lstm_cells_drop)
#   outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)

    # try basic cell

#   basic_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons)
#   outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

    # GRU cell without layers --- the best one after tuning

    gru_cells = tf.contrib.rnn.GRUCell(num_units=n_neurons) #
    outputs, states = tf.nn.dynamic_rnn(gru_cells, X, dtype=tf.float32)

    # try GRU cell with layers

#   gru_cells = [tf.contrib.rnn.GRUCell(num_units=n_neurons, activation=tf.nn.relu)
#                for layer in range(n_layers)]
#   gru_cells_drop = [tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_prob) for cell in gru_cells]
#   multi_layer_cell = tf.contrib.rnn.MultiRNNCell(gru_cells_drop)
#   outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)


with tf.name_scope("output"):
    logits = tf.layers.dense(states, n_outputs, name="logits")
    Y_prob = tf.nn.softmax(logits, name="Y_prob")

# Define the optimizer; taking as input (learning_rate) and (loss)
with tf.name_scope("loss"):
    xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_mean(xentropy, name="loss")
    loss_summary = tf.summary.scalar('log_loss', loss)

with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate) # GradientDescentOptimizer #MomentumOptimizer , momentum=0.9
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correctPrediction = tf.equal(tf.argmax(Y_prob, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))
    accuracy_summary = tf.summary.scalar('accuracy', accuracy)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# -------------------------- train model --------------------------------------


start = time.time()

with tf.Session() as sess:  # train model (train set and validation set)
    init.run()
    for epoch in range(n_epochs):

        for a_batch in batch_func(x_train_input, y_train_input, batch_size):
            iteration += 1
            (X_batch, y_batch) = a_batch
            #sess.run(training_op, feed_dict = {X:X_batch, y:y_batch,keep_prob:0.3})
            sess.run(training_op, feed_dict = {X:X_batch, y:y_batch})
        '''
        if iteration % check_interval == 0:
                loss_val = loss.eval(feed_dict={X: x_val_input, y: y_val_input})
                if loss_val < best_loss_val:
                    best_loss_val = loss_val
                    checks_since_last_progress = 0
                    best_model_params = get_model_params()
                else:
                    checks_since_last_progress += 1
        '''
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
    save_path = saver.save(sess, "./rnn_model.ckpt")
''''

        # break loop if no improvement after number of max_checks
        if checks_since_last_progress > max_checks_without_progress:
            print("Early stopping!")
            break

    # restore the parameters of the best model (the one with the lowest loss)
    if best_model_params:
        restore_model_params(best_model_params)

    acc_val_final = accuracy.eval(feed_dict={X: x_val_input, y: y_val_input})
    print("Final accuracy on validation set:", acc_val_final)
'''

print('Took: %f seconds' %(time.time() - start))

# ----------------------- make predictions ------------------------------------

# use model to predict on test set
with tf.Session() as sess:
    saver.restore(sess, "./rnn_model.ckpt") # load model parameters from disk
    Z = Y_prob.eval(feed_dict = {X: x_test})
    y_pred = np.argmax(Z, axis = 1) - 1

# print results
print(pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted']))
print(metrics.classification_report(y_test,y_pred))
print("Accuracy: " + str(metrics.accuracy_score(y_test,y_pred)))
print("Cohen's Kappa: " + str(metrics.cohen_kappa_score(y_test,y_pred)))
