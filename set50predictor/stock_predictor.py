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
    '''
        Download stock data from Yahoo Finance as listed in stock_list
        starting from start date to end date and save to CSV file.
        The default path is the same path as the running script.

        Input:
        - stock_list (String) : String of Yahoo Finance's ticker name separated by space. For example, stockA stockB stockC ...
        - start_date (String) : String of start date in format DD/MM/YYYY.
        - end_date (String) : String of end date in format DD/MM/YYYY.

        Output:
        - No return value
        - The csv file will be written with the naming convention as ticker.csv

    '''
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
      - dropout_prob (Float) : Probability to forget the date on dropout layer.

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
    '''
        Split the dataset to train and test dataset per provideed ratio.

        Input
        - ohlcv_histories (Array) : The dataset in array.
        - next_day_adj_close (Array) : The result of prediction using the dataset in array.
        - unscaled_y (Array) : The same data as next_day_adj_close but not normalize to 0-1.
        - test_split (Float) : The ratio of train per test.

        Output
        - ohlcv_train (Array) : The train dataset splitted per test_split ratio.
        - ohlcv_test (Array) : The test dataset splitted per test_split ratio.
        - y_test (Array) : The result of test dataset splitted per test_split ratio.
        - y_train (Array) : The result of train dataset splitted per test_split ratio.
        - unscaled_y_test (Array) : The unscaled y_test per test_split ratio.
    '''
    n = int(ohlcv_histories.shape[0] * test_split)

    ohlcv_train = ohlcv_histories[:n]
    y_train = next_day_adj_close[:n]

    ohlcv_test = ohlcv_histories[n:]
    y_test = next_day_adj_close[n:]

    unscaled_y_test = unscaled_y[n:]

    return ohlcv_train, ohlcv_test, y_test, y_train, unscaled_y_test

def train_predictor(ohlcv_train,y_train,ohlcv_test,y_normaliser,unscaled_y_test,hidden_layer,batch_size,epoch,dropout_probability,history_points,features_num,predict_range):
    '''
        Create LSTM model per provideded parameter, fit the train data and validate its accuracy using MSE.
        Finally, retrun the result of MSE and the model object.

        Input
        - ohlcv_train (Array) : Train dataset in array.
        - y_train (Array) : Train dataset result in array.
        - ohlcv_test (Array) : Test dataset in array.
        - y_normaliser (preprocessing.MinMaxScaler Object) : The normaliser instance that is used to scale down y_test. We will use it to scale up the result from test dataset.
        - unscaled_y_test (Array) : The unscaled y_test using for validate the result.
        - hidden_layer (Number) : LSTM parameter's number of hidden layer.
        - batch_size (Number) : LSTM parameter's number of batch size.
        - epoch (Number) : LSTM parameter's number of epoch.
        - dropout_probability (Float) : LSTM parameter's dropout probability.
        - history_points (Number) : LSTM parameter's the number of history data to train the model in each iteration.
        - features_num (Number) : LSTM parameter's the number of features in the dataset.
        - predict_range : LSTM parameter's the number of predict data to be predicted.

        Output
        - model (Object) : LSTM model which can be saved to h5 or use to predict the result with the new dataset.
        - scaled_mse (Float) : Mean Squared Error of the model measured by using the unscaled result from test dataset minus the unscaled_y_test

    '''
    model = get_LSTM_Model(hidden_layer,history_points,features_num,predict_range,'adam',dropout_probability)
    model.fit(x=ohlcv_train, y=y_train, batch_size=batch_size, epochs=epoch, shuffle=True, validation_split=0.1)

    y_test_predicted = model.predict(ohlcv_test)
    y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)

    real_mse = np.mean(np.square(unscaled_y_test - y_test_predicted))
    scaled_mse = real_mse / (np.max(unscaled_y_test) - np.min(unscaled_y_test)) * 100
    return model, scaled_mse

def train_and_validate_stock_predictor(stock,history_points,predict_range,hidden_layer,batch_size,epoch,dropout_probability, mode='file'):
    '''
        Encapsulate all activities to create LSTM model to predict the stock with parameters as provided.
        There are 2 mode for this function. mode='file' will read data from csv file and not add additional features like macd and EMA
        while other mode will read data from csv but also add macd and EMA as additional features.
        Starting from transforming data, splitting to train/test, build and fit model, evaluate the model accuracy and return result.

        Input
        - stock (String) : Ticker per Yahoo Finance.
        - history_points (Number) : LSTM parameter's the number of history data to train the model in each iteration.
        - predict_range : LSTM parameter's the number of predict data to be predicted.
        - hidden_layer (Number) : LSTM parameter's number of hidden layer.
        - batch_size (Number) : LSTM parameter's number of batch size.
        - epoch (Number) : LSTM parameter's number of epoch.
        - dropout_probability (Float) : LSTM parameter's dropout probability.

        Output
        - model (Object) : LSTM model which can be saved to h5 or use to predict the result with the new dataset.
        - scaled_mse (Float) : Mean Squared Error of the model measured by using the unscaled result from test dataset minus the unscaled_y_test
    '''
    # Read data and add MACD and EMA
    if mode=='file':
        features_num = 5
        ohlcv_histories, next_day_adj_close, unscaled_y, y_normaliser = dataset_preparation(stock+'.csv',history_points,predict_range,preprocessing.MinMaxScaler())
    else:
        df = add_macd_ema(pd.read_csv(stock+'.csv'))
        ohlcv_histories, next_day_adj_close, unscaled_y, y_normaliser = dataset_preparation('',history_points,predict_range,preprocessing.MinMaxScaler(),mode='df',df=df)
        features_num = len(df.columns)-2

    ohlcv_train, ohlcv_test, y_test, y_train, unscaled_y_test = train_test_split(ohlcv_histories,next_day_adj_close,unscaled_y)

    model, scaled_mse = train_predictor(ohlcv_train,y_train,ohlcv_test,y_normaliser,unscaled_y_test,hidden_layer,batch_size,epoch,dropout_probability,history_points,features_num,predict_range)
    return model, scaled_mse

def add_macd_ema(df,ema1=20,ema2=50):
    '''
        Compute stock technical analysis indicator - MACD and EMA and add back to the dataset

        Input
        - df (DataFrame) : The DataFrame of stock data as downloaded from Yahoo Finance
        - ema1 (Number) : The first EMA period to add to the dataset
        - ema2 (Number) : The second EMA period to add to the dataset

        Output
        - df (DataFrame) : The DataFrame with new columns added - MACD, ema1 and ema2 e.g. MACD, 20, 50 are the default column name that will be added
    '''
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
    '''
        Display the plot of the stock price in test dataset and the predicted data.

        Input
        - unscaled_y_test (Array) : The array of stock price in test dataset.
        - y_test_predicted (Array) : The array of stock price as predicted.

        Output
        - No return value.
        - The plot will be displayed on the screen.
    '''
    plt.gcf().set_size_inches(22, 15, forward=True)

    start = 0
    end = -1

    real = plt.plot(unscaled_y_test[start:end], label='real')
    pred = plt.plot(y_test_predicted[start:end], label='predicted')

    plt.legend(['Real', 'Predicted'])

    plt.show()

def train_model(stock_list, start_date, end_date):
    '''
        Initial function for the user to use by provide stock list, start and end date of data to train the prediction model.
        The function will call other function to download data from Yahoo finance per specific start and end date.
        Then, prepare the data and build LSTM model for stock price prediction of 1,5 and 10 day using the set of parameters
        that has been tuned that it can get some good result for SET50 and written all 3 models (per stock) to h5 file along with list of MSE of each model on the same path of the script.

        Input
        - stock_list (String) : List of ticker in space delimited format e.g. tickerA tickerB tickerC.
        - start_date (String) : The string of start date in format DD/MM/YYYY.
        - end_date (String) : The string of end date in format DD/MM/YYYY.

        Output
        - No return value.
        - Print the model training progress on the screen.
    '''
    
    # Split the stock_list to array
    stock_list = stock_list.split(' ')
    
    # Convert string to datetime object
    start_date = datetime.datetime.strptime(start_date,"%d/%m/%Y")
    end_date = datetime.datetime.strptime(end_date,"%d/%m/%Y")

    try:
        get_stock_data(stock_list, start_date, end_date)
    except Exception as e:
        print("exception "+str(e)+"on "+stock_list)
    
    # Array for recording MSE from each round of training
    mse_list = [] 
    
    # Train model to predict 1, 5 and 10 days using the best parameters from SET50 as found
    # Save the trained model to h5 format to be used by query price function
    for stock in stock_list:

        print('start model training for stock = '+stock+'. It may take at least 5 minutes...')

        # Train model for 1 day predict range by using parameters that we found from our study earlier

        # Set up parameters for the model
        predict_range = 1
        history_points = 90
        hidden_layer = 10
        batch_size = 10
        epoch = 90
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

        # Train model for 5 days predict range by using parameters that we found from our study earlier

        # Set up parameters for the model
        predict_range = 5
        history_points = 30
        hidden_layer = 70
        batch_size = 10
        epoch = 60
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

        # Train model for 10 days predict range by using parameters that we found from our study earlier
        predict_range = 10
        history_points = 50
        hidden_layer = 60
        batch_size = 10
        epoch = 80
        dropout_probability = 0.3
        mode = 'file'

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
    '''
        Query the predicted price from the model of the stocks that were trained by providing the stock name and the date range of prediction from the end date of training data.
        It supports only 1,5 and 10 date range.

        Input
        - stock_list (String) : List of ticker in space delimited format e.g. tickerA tickerB tickerC.
        - date_range (Number) : The number of date range to predict the price - 1, 5 and 10 days from end date of the training data set.
    '''
    
    # Split the stock list to array
    stock_list = stock_list.split(' ')
    
    # Read MSE of the trained model for each stock and each range of prediction
    df_mse = pd.read_csv('mse_list.csv')
    
    if date_range <=10:
        for stock in stock_list:
            
            try:
                
                # Do prediction by using history_points of data as we found from our study earlier
                if date_range == 1:
                    predict_range = 1
                    history_points = 90
                    mode = 'file'
                if date_range <= 5:
                    predict_range = 5
                    history_points = 30
                    mode = 'file'
                if date_range <= 10:
                    predict_range = 50
                    history_points = 90
                    mode = 'df'
                
                model = load_model(stock+'_'+str(predict_range)+'.h5')
                
                df_stock = pd.read_csv(stock+'.csv')
                df_stock = add_macd_ema(df_stock)

                ohlcv_histories, next_day_adj_close, unscaled_y, y_normaliser = dataset_preparation(stock+'.csv',history_points,predict_range,preprocessing.MinMaxScaler(),mode=mode,df=df_stock)

                adj_predicted = model.predict(ohlcv_histories[len(ohlcv_histories)-1:])
                adj_predicted = y_normaliser.inverse_transform(adj_predicted)

                print(stock+' price prediction for '+str(date_range)+' days : '+str(adj_predicted[0][:date_range]))
                print("Mean square error = "+str(df_mse[(df_mse['0']==stock) & (df_mse['1']==predict_range)]['2'].values)+" %")
            except Exception as e:
                print("There was an error : "+str(e))
                continue
                
    else:
        print("Support prediction from the end date of training data for 1 to 10 days only. Please try again.")
        
    
            

