from pandas import DataFrame
from models import lstm 

import yfinance as yf
import matplotlib.pyplot as plt
import math
import numpy as np

def prep_data(data:DataFrame, train_fraction:float, labels:list, train_history:int):

    # get initial training data set 
    data = data.loc[:,labels]
    # print(data)
    train_length = math.ceil(len(data)*train_fraction)
    train_data = data.iloc[:train_length,:]
    test_data = data.iloc[train_length:,:]

    # loop thru to create train data
    #TODO: prep test data at the same time
    # !this is slow.... there is probably a reshape() somewhere in numpy or pandas that is better than this loop...

    steps = train_history
    train_y = []
    train_x = []
    for i in range(steps, len(train_data)-1):
        train_y.append(train_data.iloc[i+1,0])
        data = []

        for j in range(i-60,i):
            add = list(train_data.iloc[j])
            data.append(add)
        train_x.append(data)

    print(np.shape(train_x))
    print(np.shape(train_y))

    #TODO: normalise data before returning        
    np_train_x, np_train_y = np.array(train_x, np.float32), np.array(train_y, np.float32)
    # print(np_train_x)
    return np_train_x, np_train_y


def main():
    stock = yf.Ticker('CBA.AX')
    info_labels = ['symbol','sector','exchange','fullTimeEmployees']
    # info = {key : stock.info.get(key) for key in info_labels}
    ticker = stock.info.get('symbol')
    history = stock.history(period='12mo')
    history['Ticker'] = ticker

    print(history)
    training_length = math.ceil(len(history)*0.7)
    train_data = history.iloc[:training_length,:]

    columns = ['Close']            

    train_x, train_y = prep_data(history, 0.7, columns, 60)

    lstm(train_x, train_y)
    
    plot = history.plot(y='Close')
    plt.show()
    

if __name__ == "__main__":
    main()
