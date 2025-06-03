import pandas as pd


class DataUtils:
    def load_news_data(self, ticker = None):  

        news = pd.read_csv(
            "../data/raw_analyst_ratings.csv",
            parse_dates=["date"],
            index_col=["date"],
        )
        news.drop("Unnamed: 0", axis=1, inplace=True)
        news.index = pd.to_datetime(news.index, utc=True, format="ISO8601")

        if ticker:
            news = news.loc[news['stock'] == ticker]
            return news
        else:
            return news
        

    def load_historical_data(self, ticker: str):
        ticker_data = {
            "AAPL": "AAPL_historical_data",
            "AMZN": "AMZN_historical_data",
            "GOOG": "GOOG_historical_data",
            "META": "META_historical_data",
            "MSFT": "MSFT_historical_data",
            "NVDA": "NVDA_historical_data",
            "TSLA": "TSLA_historical_data",
        }

        hist_data = pd.read_csv(
            f"../data/yfinance_data/{ticker_data[ticker]}.csv",
    
        ).set_index('Date')
        hist_data.index = pd.to_datetime(hist_data.index)

        hist_data.drop(['Dividends', 'Stock Splits'], axis=1, inplace=True)

        return hist_data
    
    def extract_domain(self, email):
        return email.split('@')[1]