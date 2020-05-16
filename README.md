# Building & Tuning LSTM to predict stock price in SET50
## Background
It is always be better to know the future especially in stock trading as we can plan well on when to buy and when to sell to make profit. 

As I search over the internet, I found some example of using deep-learning algorithms like Long Short-Term Memory (LSTM) to predict stock price. 

However, I haven't seen much example and result of tuning LSTM for different range of prediction to get the best result and also the accuracy of the model when we use it with various number of stock.

## Objectives
1. Build a stock predictor using LSTM and tune the paramters for one selected stock in predicting its Adjusted Close price in the next 1, 5 and 10 days. The way that I intend to tune the parameters are:
  - Starting by using the lowest value for all parameters and allow only one parameter to be adjusted.
  - Loop thru the cycle of build, train and validate different value of the parameters to find the best value of that parameters. 
  - Do this for all parameters to see which parameter and which value give the lowest error.
  - Update the particular parameter with value while the others is still the lowest.
  - Repeat all steps until the error doesn't get lower and that will be the set of best parameter value.

2. Build model by using the set of parameters as found from previous topics to do prediction on another set of stock and measure how many stock that the model could predict with acceptable error rate.

3. Build user-friendly script that the user is able to:
  - Provide list of stocks, date range of historical data and train the model.
  - Query the predicted price by selecting the stock and the date range of price prediction.

## How to run this project
This project concists of 2 Jupyter Notebooks:
- `stock_predictor_tuning_study.ipynb` is the Jupyter Notebook that I use it to explain each steps of my work as stated in the objectives in detail.
I recommend to go thru it to see the study and code in details. It covers the objectives 1 and 2 in this notebook.

- `stock_predictor.ipynb` is another Jupyter Notebook that I encapsulate all of my functions into 2 simple functions where other interested person can try train the predictor for their interested stocks and get the prediction.

### How to run stock_predictor_tuning_study
1. I strongly recommended to run my notebook in Google colaboratory - https://colab.research.google.com as the notebook requires the machine that support tensorflow and this online editor already provides everything ready to use.
So, download it to your computer first.
2. At colaboratory, go to file > upload notebook and select `stock_predictor_tuning_stidy.ipynb`.
3. At Files panel, click **mount Google Drive** and follow the on screen instruction. After the mounting process is finished, you will see the folder `Colab Notebooks` created.
Create a folder under `Colab Notebooks` named **Data**.
4. You should be good to go thru and run each code cell in the notebook.

### How to run stock_predictor
1. Download `stock_predictor.ipynb` and `stock_predictor.py`
2. Go to https://colab.research.google.com and go to file and upload new notebook. Upload stock_predictor.ipynb.
3. Upload `stock_predictor.py` to Files panel by drag and drop from your local computer to the root/outmost level folder.
4. Follow how to use to do train the model and do prediction.

## Result Summary
1. One way to tune the parameter of LSTM is to iteratively adjust one by one by starting from setting every parameter at lowest and iteratively change only one parameter at a time to find the value that has lowest MSE. Then, use that parameter value along with other parameters and iterate to find the best value of the next parameter.

2. The result of parameters that we get base on using the data of open, high, low, volume to predict adj. close can be used to predict other stocks as well. If we use 5.5% MSE as acceptable error rate we can use this parameters to predict 40% of stock in SET50 for 1 day prediction, 20% of stock for 5 days prediction and about 5% of stock for 10 days prediction.

3. The longer range of prediction the higher error that we will get.

4. Adding technical analysis indicator like MACD and EMA does help improve the prediction accuracy for 50% of the stock in 5 days prediction.

5. At the end of the day, to optimize accuracy of the prediction we have to do it stock by stock as the fetures that will effect is varies.
