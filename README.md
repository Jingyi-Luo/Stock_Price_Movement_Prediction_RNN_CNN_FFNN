# Stock Price Movement Prediction Using FFNN CNN RNN

This project aims to predict the movement of future trading price of Netflix (NFLX) stock using transaction data on January 3, 2017 from the Limit Order Book (LOB). Stationary features were created to overcome autocorrelation and reduce noises of the time series data. For this project, a random forest model was built as baseline and three types of neural network models were develpted and compared: Feed Forward Neural Networks (FFNN), Convolutional Neural Networks (CNN), and Recurrent Neural Networks (RNN).  The models were compared based on not only the accuracy, but also other metrics such as recall and Cohen’s Kappa.

## Data, Feature Engineering and Labelling

**Data**

The raw data for this project is the high frequency stock market data, which contains information on all the sell and buy transactions for stocks traded on NASDAQ on January 3, 2017. Example fields included in the data are timestamp, order id, book event type, price, quantity and side. For this project, the data for NFLX are used. There are 477,239 observations, including Add, Modify, Trade and Cancel event types. To predict future movement of trading price, only traded events are included which has 19,599 observations. This data is splitted into train and test sets at a ratio of 7:3 from the time horizon due to the time-series nature of the data. Table 1 shows the details about the structures for each set and the labeling process will be explained later. 

<img width="665" alt="table1_train_test_sets_structure" src="https://user-images.githubusercontent.com/42804316/57708270-0cb55500-7637-11e9-86d5-0ca02547a17c.png">

**Feature Engineering**

To mitigate autocorrelation and reduce noise in time series data, several features were created from existing features. For instance, a mixed moving average was calculated using moving averages in the windows of 3, 6, 12, and 24. Other features created include the count of rising prices among the past 10 traded transactions, average quantity for the past transactions and price difference between every other 5 transactions.  See Table 2 for details about each feature used in the models. 

<img width="797" alt="table2_feature_engineering" src="https://user-images.githubusercontent.com/42804316/57708579-949b5f00-7637-11e9-8129-046e222a7efe.png">

Then, the numerical features were standardized with a mean of 0 and a standard deviation of 1 to ensure they are treated the same in the models regardless of their units.. Lastly, to ensure the input data is consistent for different types of models, the train and test sets were reshaped such that one instance (or one row in the data frame) corresponded to all 50 transactions in one prediction window. For instance, we would have 400 features (8 times 50) at transaction T49 for predicting the price for transaction T50.

**Labelling**

Ground truth labels were generated to indicate the movement of future trading prices. A window size of 50 was defined for the current transaction. Specifically, at transaction T49, we are interested in predicting the price for transaction T50, with T denoting transaction. The average price for the transactions within the prediction window (avg_PT0 to T49)  is compared with the price for the transaction right after (PT50). To reduce noise, we determined a threshold above which a price change would be labeled as an upward or downward change. A threshold of 0.03% was chosen to ensure the classes are balanced in the train set. See Table 3 for more details about the labeling process. 

<img width="771" alt="table3_labelling" src="https://user-images.githubusercontent.com/42804316/57709292-ba753380-7638-11e9-8c69-c8d09bd5a65c.png">

## Architectures of Neural Network Models

**Feed Forward Neural Networks (FFNN)**

The best FFNN has two hidden layers with ReLU as the activation functions. The first hidden layer has 300 neurons and the second layer has 100 neurons. The output layer outputs the logits and then goes through the softmax activation function. The cross entropy is utilized as the loss function and adam optimizer is adopted to optimized the parameters (weights and bias) of the model.. The network takes 400 features as the number of inputs for each instance, and outputs 3 probabilities for each class. The predicted class is determined by the highest probability among the three. During training session, the entire train set was splitted into train and validation sets sequentially at a ratio of 8:2. Specifically, the first 10,935 instances were in the train set while the remaining 2,734 instances were in the validation set. 

<img width="60%" alt="tensorboard_graph" src="https://user-images.githubusercontent.com/42804316/57709888-d927fa00-7639-11e9-967e-7ba0e0046907.png"><img width="40%" alt="simplified_flowchart" src="https://user-images.githubusercontent.com/42804316/57709910-e1803500-7639-11e9-9d7b-6936507a54b2.png"><br /> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Fig 1. Tensorboard Computational Graph (left) and Simplified Flowchart (right)

**Convolutional Neural Networks (CNN)**

It consists of one input layer, two convolutional layers, one pooling layers, one fully connected layer accompanied by one dropout layer,  and one output layer. The number of inputs is 400 which is the product of the height (50) and width (8) of one grid. Then, the input is further reshaped to accommodate the 2-D convolution which has a height of 50, a width of 8 and channels of 1. For two convolutional layers, 36 filters and 72 filters are utilized, respectively. For both layers, the kernel size is two, stride is one, padding is SAME, Relu is  the activation function. In the max pooling, the dimensionality of the feature maps are reduced to the half both for height and width using a kernel of two by two and a stride of two by two. The outputs from the pooling layer are reshaped back to 1-d vector for the fully connected layer, after which the dropout layer (dropout rate: 0.5) is applied to decrease overfitting. Finally, the cross entropy and the Adam optimizer are used to calculate the loss and train the model. 

<img width="40%" alt="tensorboard_graph" src="https://user-images.githubusercontent.com/42804316/57710492-0a54fa00-763b-11e9-972b-8c6598f4b95c.png"><img width="40%" alt="simplified_graph" src="https://user-images.githubusercontent.com/42804316/57710502-117c0800-763b-11e9-8b6f-39b4b38eaee7.png"><br /> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Fig 1. Tensorboard Computational Graph (left) and Simplified Flowchart (right)


**Recurrent Neural Networks (RNN)**



## Results

