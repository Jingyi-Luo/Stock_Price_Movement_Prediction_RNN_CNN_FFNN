# Stock Price Movement Prediction Using FFNN CNN RNN

This project aims to predict the movement of future trading price of Netflix (NFLX) stock using transaction data on January 3, 2017 from the Limit Order Book (LOB). Stationary features were created to overcome autocorrelation and reduce noises of the time series data. For this project, a random forest model was built as baseline and three types of neural network models were develpted and compared: Feed Forward Neural Networks (FFNN), Convolutional Neural Networks (CNN), and Recurrent Neural Networks (RNN).  

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



## Results

