'''
How to get the dataset of one company stock transaction in one day.

# timestamp :  High resolution timestamp for this event
# order id:  Unique id for the order (Added, Modified,...)
# book event type:  Add, Modify, Cancel, Trade
# price: Price of Add, Modify, Trade
# quantity: Quantity of Add, Modify, Trade
# aux quantity: Auxiliary quantity info (e.g. Amount left on order after partial trade)
# side: Side of event (Bid vs Ask)
# aux1 Additional info - e.g. trade condition
# aux2 Additional info - e.g. sales condition
'''

import pandas as pd
import os

path = '/PATH/Healthy market data/Book_event_levelII'
os.chdir(path)
reread = pd.HDFStore('book_events_total_view_2017-01-03.h5')

# This h5 file contains ALL transaction with ALL the Nasdaq companies at 2017-01-03.
# Here's how you can get different comanies for this file and change it to CSV file.
i=0
for key in reread:
    # print(key, reread[key].shape, sep = ',', end= '\n')
    i+=1
print(i)
#count of companies in Nasdaq is == 8333

#let's do this only on one key.. Bank of America... /BAC
print(reread['/BAC'].shape) #(757503, 9)
# print(list(reread['/BAC'].columns.values)) #['timestamp', 'order_id', 'book_event_type', 'price', 'quantity', 'aux_quantity', 'side', 'aux1', 'aux2']
# print(reread['/BAC'].head(2))

BAC = reread['/BAC'][['timestamp' ,'order_id', 'book_event_type', 'price', 'quantity', 'side']]

print(BAC.shape) #(757503, 6)

BAC.to_csv('BAC_book_events_sip-nyse_2016-05-02.csv', sep=',')
