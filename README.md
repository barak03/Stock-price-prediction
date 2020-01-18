# Stock-price-prediction
a simpale script in python that gets stock price data from alpha vantage free API and then predicts thr price for each minute in the tarde
day using Random Forest Regressor.\
Basically every price we predicted is stored in the list so to convert the indexes back to the time I wrote a dictionary that every key is a list index and every value is the exact minute in the trading day.
Initially, the script checks from a list of stock symbols (that we pre-wrote it) which stocks to trade based on the closing price on the last trading day and.\
Then for each share it pulls price data from alpha vantage API and predicts its price for every minute in the day,\
and for each share it checks when there is a sequence Price increase to know when to buy and when to sell it.\
The script works according to the New York Clock and can identify when the stock trading is open and when it is closed.

### Example for identifying a day when stock trading is closed (Saturday)
![closed trading tay](https://github.com/barak03/Stock-price-prediction/blob/master/images/closed%20trading%20day.png)
