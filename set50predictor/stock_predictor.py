import yfinance as yf
import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from keras import optimizers
import numpy as np
from keras.models import Sequential
from keras.utils import plot_model
from keras.models import load_model

# Yahoo Finance Library for getting historical stock data from Yahoo finance
import os.path, time
import datetime

def get_stock_data(stock_list,start_date,end_date):
  for stock in stock_list:
    try:
      file = stock+'.csv'
      print('Downloading data from Yahoo Finance...')
      data = yf.download(stock, start=start_date, end=end_date)
      pd.DataFrame(data).to_csv(file)
    except Exception as e:
      print("exception "+str(e)+"on "+stock)
      print("start date = "+start_date)
      print("end date = "+end_date)

def dataset_preparation(filename, history_points, predict_range, y_normaliser, mode='file', df=None):
    '''
      This function will prepare data and make it ready for training and testing the model by receiving the CSV file path or dataframe 
      equivalent to Yahoo Finance's historical data with other parameters, normalize to 0-1 range value, separate into train and test and return the result.

      Input:
      - filename (String) : The file path for csv file containing historical stock data downloaded from Yahoo Finance. 
      - history_points (Number) : The number of day range for historical data to be used for training the model.
      - predict_range (Number) : The range of day to forecast the price.
      - y_normmalizer (preprocessing.MinMaxScaler Object) :  Preprocessor for normalize the price for forecast data to be 0-1. We need this so that we could use it again for scaling up the data back to normal value.
      - df (DataFrame) : The dataframe input of the dataset. If the filename is passed and the mode is set to'file' it will be ignored.
      - mode : If it is 'file' the function will ignore df otherwise it will assume that df will be provided for preparation. This is used in the case that the data is read from CSV and append with other data features.

      Output:
      - ohlcv_histories_normalised (Array) : Array of the data features. One row consist of [day-1-open,day-1-max,day-1-min,...day-history_point ].
      - next_day_adjclose_values_normalised (Array) : Array of normalised Adj Close values transformed in the format of [day1-adj close,day2-adj close....day-predict_range adj close].
      - next_day_adjclose_values (Array) : Array of actual Adj Close values transformed in the format of [day1-adj close,day2-adj close....day-predict_range adj close].
      - y_normaliser (preprocessing.MinMaxScaler Object) : After we fit the actual value, we return it back so that it can be used again to scale the normalized result.
    '''

    # Prepare data per mode - file or dataframe input
    if mode=='file':
      # If it is file mode the function expect CSV file path to read the data
      df = pd.read_csv(filename)

    # For both mode, we will drop row with null value as we can't use it anyway
    df_na = df.dropna(axis=0)
    # Drop Date as this is time series data, Date isn't used. Also drop Close as we will predict Adj Close.
    df_na = df_na.drop(['Date','Close'],axis=1)

    # Normalise all data to the value range of 0-1 as neural network algorithm has better performance with this data range
    data_normaliser = preprocessing.MinMaxScaler()
    data_normalised = data_normaliser.fit_transform(df_na)

    # Prepare the data in the format of [day-1-open,day-1-max,day-1-min,...day-history_point ] as 1 row input for predict the 'predict_range' price for train and test
    ohlcv_histories_normalised = np.array([data_normalised[i  : i + history_points].copy() for i in range(len(data_normalised) - history_points - predict_range +1)])

    # Get the actual price [day1-adj close,day2-adj close....day-predict_range adj close] for train and test
    next_day_adjclose_values_normalised = np.array([data_normalised[i + history_points:i + history_points + predict_range,3].copy() for i in range(len(data_normalised) - history_points - predict_range+1)])

    # Create the same array as the normalised adj close but with the actual value not the scaled down value. This is used to calculate the prediction accuracy
    next_day_adjclose_values = np.array([df_na.iloc[i + history_points:i + history_points+predict_range]['Adj Close'].values.copy() for i in range(len(df_na) - history_points - predict_range+1)])    

    # Use the passed normaliser to fit the actual value so that we can scale the predicted result back to actual value
    y_normaliser.fit(next_day_adjclose_values)

    return ohlcv_histories_normalised, next_day_adjclose_values_normalised, next_day_adjclose_values, y_normaliser

def get_LSTM_Model(layer_num, history_points, features_num,predict_range,optimizer,dropout_prob):

    '''
      This function will build LSTM model per provided parameters.
      The model will be a simple one consist of one forget layer with configurable forget probability and configurable number of hidden layers.

      Input:
      - layer_num (Number) : The number of hidden layer.
      - history_points (Number) : The number of data in the dataset.
      - features_num (Number) : The number of features in the dataset.
      - predict_range (Number) : The number of day to predict the stock price.
      - optimizer (Number) : The optimizer's name e.g. adam.
      - dropout_prob (float) : Probability to forget the date on dropout layer.

      Output:
      - model (Object) : The compiled LSTM model per the provided parameters.

    '''

    # Initialize LSTM using Keras library
    model = Sequential()
    # Defining hidden layer number and the shape of the input (number of data in the dataset and the number of feature)
    model.add(LSTM(layer_num, input_shape=(history_points, features_num)))

    # Add forget (dropout) layer with probability per argument
    model.add(Dropout(dropout_prob))

    # End the network with hiddenlayer per the size of forecast day e.g. 1,5,10
    model.add(Dense(predict_range))

    # Build and return the model per the selected optimizer
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

def train_test_split(ohlcv_histories, next_day_adj_close, unscaled_y,test_split = 0.9):
    n = int(ohlcv_histories.shape[0] * test_split)

    ohlcv_train = ohlcv_histories[:n]
    y_train = next_day_adj_close[:n]

    ohlcv_test = ohlcv_histories[n:]
    y_test = next_day_adj_close[n:]

    unscaled_y_test = unscaled_y[n:]

    return ohlcv_train, ohlcv_test, y_test, y_train, unscaled_y_test

def train_predictor(ohlcv_train,y_train,ohlcv_test,y_normaliser,unscaled_y_test,hidden_layer,batch_size,epoch,dropout_probability,history_points,features_num,predict_range):
    model = get_LSTM_Model(hidden_layer,history_points,features_num,predict_range,'adam',dropout_probability)
    model.fit(x=ohlcv_train, y=y_train, batch_size=batch_size, epochs=epoch, shuffle=True, validation_split=0.1,verbose=0)

    y_test_predicted = model.predict(ohlcv_test)
    y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)

    real_mse = np.mean(np.square(unscaled_y_test - y_test_predicted))
    scaled_mse = real_mse / (np.max(unscaled_y_test) - np.min(unscaled_y_test)) * 100
    return model, scaled_mse

def train_and_validate_stock_predictor(stock,history_points,predict_range,hidden_layer,batch_size,epoch,dropout_probability, mode='file'):

    # Read data and add MACD and EMA
    print('Preparing Data...')
    if mode=='file':
        features_num = 5
        ohlcv_histories, next_day_adj_close, unscaled_y, y_normaliser = dataset_preparation(stock+'.csv',history_points,predict_range,preprocessing.MinMaxScaler())
    else:
        df = add_macd_ema(pd.read_csv(stock+'.csv'))
        ohlcv_histories, next_day_adj_close, unscaled_y, y_normaliser = dataset_preparation('',history_points,predict_range,preprocessing.MinMaxScaler(),mode='df',df=df)
        features_num = len(df.columns)-2

    ohlcv_train, ohlcv_test, y_test, y_train, unscaled_y_test = train_test_split(ohlcv_histories,next_day_adj_close,unscaled_y)

    print('Train the model...')
    model, scaled_mse = train_predictor(ohlcv_train,y_train,ohlcv_test,y_normaliser,unscaled_y_test,hidden_layer,batch_size,epoch,dropout_probability,history_points,features_num,predict_range)
    return model, scaled_mse

def add_macd_ema(df,ema1=20,ema2=50):

    df_close = df[['Close']]
    df_close.reset_index(level=0, inplace=True)
    df_close.columns=['ds','y']

    exp1 = df_close.y.ewm(span=12, adjust=False).mean()
    exp2 = df_close.y.ewm(span=26, adjust=False).mean()
    macd = exp1-exp2

    df = pd.merge(df,macd,how='left',left_on=None, right_on=None, left_index=True, right_index=True)
    df.columns = ['Date','Open','High','Low','Close','Adj Close','Volume','MACD']
    df[ema1] = df['Close'].ewm(span=ema1, adjust=False).mean()
    df[ema2] = df['Close'].ewm(span=ema2, adjust=False).mean()

    return df

def display_test_validation_graph(unscaled_y_test, y_test_predicted):
    # Display graph test data only

    plt.gcf().set_size_inches(22, 15, forward=True)

    start = 0
    end = -1

    real = plt.plot(unscaled_y_test[start:end], label='real')
    pred = plt.plot(y_test_predicted[start:end], label='predicted')

    plt.legend(['Real', 'Predicted'])

    plt.show()

def train_model(stock_list, start_date, end_date):

    stock_list = stock_list.split(' ')
    start_date = datetime.datetime.strptime(start_date,"%d/%m/%Y")
    end_date = datetime.datetime.strptime(end_date,"%d/%m/%Y")

    get_stock_data(stock_list, start_date, end_date)
    mse_list = []

    for stock in stock_list:

        print('start model training for stock = '+stock+'. It may take at least 5 minutes...')

        # Train model for 1 day predict range

        # Set up parameters for the model
        predict_range = 1
        history_points = 30
        hidden_layer = 80
        batch_size = 10
        epoch = 30
        dropout_probability = 1.0
        mode = 'file'

        try:
            model, scaled_mse = train_and_validate_stock_predictor(stock,history_points,predict_range,hidden_layer,batch_size,epoch,dropout_probability,mode)
            mse_list.append([stock, predict_range, scaled_mse])
            model.save(stock+'_'+str(predict_range)+'.h5')
        except Exception as e:
            print("exception "+str(e)+"on "+stock)
            pd.DataFrame(columns=['predict rage','stock','exception'],data=[predict_range,stock,str(e)]).to_csv('exception.csv')
            continue

        # Train model for 5 days predict range

        # Set up parameters for the model
        predict_range = 5
        history_points = 30
        hidden_layer = 80
        batch_size = 20
        epoch = 30
        dropout_probability = 0.1
        mode = 'file'

        try:
            model, scaled_mse = train_and_validate_stock_predictor(stock,history_points,predict_range,hidden_layer,batch_size,epoch,dropout_probability,mode)
            mse_list.append([stock, predict_range, scaled_mse])
            model.save(stock+'_'+str(predict_range)+'.h5')
        except Exception as e:
            print("exception "+str(e)+"on "+stock)
            pd.DataFrame(columns=['predict rage','stock','exception'],data=[predict_range,stock,str(e)]).to_csv('exception.csv')
            continue

        # Train model for 10 days predict range
        predict_range = 10
        history_points = 90
        hidden_layer = 80
        batch_size = 20
        epoch = 30
        dropout_probability = 0.1
        mode = 'df'

        try:
            model, scaled_mse = train_and_validate_stock_predictor(stock,history_points,predict_range,hidden_layer,batch_size,epoch,dropout_probability,mode)
            print("Predict {} days for {} with MSE = {}".format(str(predict_range),str(stock),str(scaled_mse)))
            mse_list.append([stock, predict_range, scaled_mse])
            model.save(stock+'_'+str(predict_range)+'.h5')
        except Exception as e:
            print("exception "+str(e)+"on "+stock)
            pd.DataFrame(columns=['predict rage','stock','exception'],data=[predict_range,stock,str(e)]).to_csv('exception.csv')
            continue

    print("Completed...")
    pd.DataFrame(mse_list).to_csv('mse_list.csv')

def query_price(stock_list,date_range):

    _home = '/content/drive/My Drive/Colab Notebooks/'
    stock_list = stock_list.split(' ')
    df_mse = pd.read_csv(_home+'mse_list.csv')

    for stock in stock_list:

        if date_range in [1,5,10]:
            model = load_model(_homestock+'_'+str(date_range)+'.h5')

            # Do prediction
            if date_range == 1:
                predict_range = 1
                history_points = 30
                mode = 'file'
            if date_range == 5:
                predict_range = 5
                history_points = 30
                mode = 'file'
            if date_range == 10:
                predict_range = 10
                history_points = 90
                mode = 'df'

            df_stock = pd.read_csv(stock+'.csv')
            df_stock = add_macd_ema(df_stock)

            ohlcv_histories, next_day_adj_close, unscaled_y, y_normaliser = dataset_preparation(stock+'.csv',history_points,predict_range,preprocessing.MinMaxScaler(),mode=mode,df=df_stock)

            adj_predicted = model.predict(ohlcv_histories[len(ohlcv_histories)-1:])
            adj_predicted = y_normaliser.inverse_transform(adj_predicted)

    print(stock+' price prediction for '+str(date_range)+' days : '+str(adj_predicted[0]))
    print("Mean square error = "+str(df_mse[(df_mse['0']==stock) & (df_mse['1']==predict_range)]['2'].values)+" %")

