import talib as ta
import pandas as pd
import matplotlib.pyplot as plt

from utils import DataUtils 

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt import plotting
import cvxpy

class TechnicalIndicator:
    def __init__(self, data):
        self.data = data

    def lagging_indicators(self):
        '''
        Lagging indicators are used to confirm trends and often follow the price movement. 
        '''
        self.data['EMA'] = ta.EMA(self.data['Close'], timeperiod=30)
        self.data['WMA'] = ta.WMA(self.data['Close'], timeperiod=30)
        self.data['ADXR'] = ta.ADXR(self.data['High'], self.data['Low'], self.data['Close'], timeperiod=14)
        self.data['RSI'] = ta.RSI(self.data['Close'], timeperiod=14)
        self.data['macd'], self.data['macdsignal'], self.data['macdhist'] = ta.MACD(self.data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)

    def leading_indicators(self):
        '''
        Leading indicators are used to predict future price movements. 
        '''
        self.data['CMO'] = ta.CMO(self.data['Close'], timeperiod=14)
        self.data['TSF'] = ta.TSF(self.data['Close'], timeperiod=14)

    def instantaneous_indicators(self):
        '''
        Instantaneous indicators provide a snapshot of the market condition at a specific moment, without considering past data. 
        '''
        self.data['HT_TRENDLINE'] = ta.HT_TRENDLINE(self.data['Close'])
        self.data['TYPPRICE'] = ta.TYPPRICE(self.data['High'], self.data['Low'], self.data['Close'])
        self.data['HT_TRENDMODE'] = ta.HT_TRENDMODE(self.data['Close'])

    def volume_indicators(self):
        '''
        Volume indicators are used to analyze trading volume and its relationship with price movements.
        '''
        self.data['MFI'] = ta.MFI(self.data['High'], self.data['Low'], self.data['Close'], self.data['Volume'], timeperiod=14)
        self.data['OBV'] = ta.OBV(self.data['Close'], self.data['Volume'])
        self.data['AD'] = ta.AD(self.data['High'], self.data['Low'], self.data['Close'], self.data['Volume'])

    def volatility_indicators(self):
        '''
        Volatility indicators measure the degree of variation in a financial instrument's price over time.
        '''
        # Bollinger band
        self.data['BB_upperband'], self.data['BB_middleband'], self.data['BB_lowerband'] = ta.BBANDS(self.data['Close'], timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
        self.data['ATR'] = ta.ATR(self.data['High'], self.data['Low'], self.data['Close'], timeperiod=14)
        self.data['NATR'] = ta.NATR(self.data['High'], self.data['Low'], self.data['Close'], timeperiod=14)


    def plot_price_indicators(self, data: pd.DataFrame):
        '''
        Plots indicators that are close to the price
        '''

        fig, ax1 = plt.subplots(figsize=(14, 7))

        # Plot the Close price
        ax1.plot(data.index, data['Close'], label='Close Price', color='blue')

        # Plot the indicators
        ax1.plot(data.index, data['EMA'], label='EMA', color='red')
        ax1.plot(data.index, data['WMA'], label='WMA', color='green')
        ax1.plot(data.index, data['HT_TRENDLINE'], label='HT_TRENDLINE', color='brown')
        ax1.plot(data.index, data['TYPPRICE'], label='TYPPRICE', color='pink')
        ax1.plot(data.index, data['TSF'], label='TSF', color='cyan')

        ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))

        ax1.set_title('Close Price with Indicators')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Value')

        plt.tight_layout()
        plt.show()

    
    def plot_RSI(self, data):
        fig, axes = plt.subplots(nrows=2, figsize=(14, 7))

        # Plot the Close price
        axes[0].plot(data.index, data['Close'], label='Close Price', color='blue')
        axes[0].set_title("Closing price")

        axes[1].plot(data.index, data['RSI'], label='RSI', color='purple')
        axes[1].axhline(y=20, color='red', linestyle='--', label='Oversold (20)')
        axes[1].axhline(y=80, color='green', linestyle='--', label='Overbought (80)')
        
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

        axes[1].set_title('RSI Indicators with 20 and 80 as oversold and overbought lines')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Value')

        plt.tight_layout()
        plt.show()

    def plot_CMO(self, data):
        fig, axes = plt.subplots(nrows=2, figsize=(14, 7))

        # Plot the Close price
        axes[0].plot(data.index, data['Close'], label='Close Price', color='blue')

        axes[1].plot(data.index, data['CMO'], label='CMO', color='gray')
        axes[1].axhline(y=-50, color='red', linestyle='--', label='Oversold (-50)')
        axes[1].axhline(y=50, color='green', linestyle='--', label='Overbought (50)')

        axes[0].set_title("Closing price")
        
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

        axes[1].set_title('CMO Indicator')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Value')

        plt.tight_layout()
        plt.show()

    def plot_MFI(self, data):

        fig, axes = plt.subplots(nrows=2, figsize=(14, 7))

        # Plot the Close price
        axes[0].plot(data.index, data['Close'], label='Close Price', color='blue')
        axes[0].set_title("Closing price")

        axes[1].plot(data.index, data['MFI'], label='MFI', color='magenta')
        axes[1].axhline(y=20, color='red', linestyle='--', label='Oversold (20)')
        axes[1].axhline(y=80, color='green', linestyle='--', label='Overbought (80)')

        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

        axes[1].set_title('MFI Indicator')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Value')

        plt.tight_layout()
        plt.show()

    def plot_OBV(self, data):

        fig, axes = plt.subplots(nrows=2, figsize=(14, 7))

        # Plot the Close price
        axes[0].plot(data.index, data['Close'], label='Close Price', color='blue')
        axes[0].set_title("Closing price")

        axes[1].plot(data.index, data['OBV'], label='OBV', color='yellow')

        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

        axes[1].set_title('OBV Indicator')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Value')

        plt.tight_layout()
        plt.show()


    def plot_AD(self, data):

        fig, axes = plt.subplots(nrows=2, figsize=(14, 7))

        # Plot the Close price
        axes[0].plot(data.index, data['Close'], label='Close Price', color='blue')
        axes[0].set_title("Closing price")

        axes[1].plot(data.index, data['AD'], label='AD', color='darkblue')

        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

        axes[1].set_title('AD Indicator')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Value')

        plt.tight_layout()
        plt.show()     


    def plot_ATR(self, data):

        fig, axes = plt.subplots(nrows=2, figsize=(14, 7))

        # Plot the Close price
        axes[0].plot(data.index, data['Close'], label='Close Price', color='blue')
        axes[0].set_title("Closing price")

        axes[1].plot(data.index, data['ATR'], label='ATR', color='darkgreen')

        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

        axes[1].set_title('ATR Indicator')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Value')

        plt.tight_layout()
        plt.show()                  

    def plot_ADXR(self, data):

        fig, axes = plt.subplots(nrows=2, figsize=(14, 7))

        # Plot the Close price
        axes[0].plot(data.index, data['Close'], label='Close Price', color='blue')
        axes[0].set_title("Closing price")

        axes[1].plot(data.index, data['ADXR'], label='ADXR', color='yellow')
        axes[1].axhline(y=20, color='red', linestyle='--', label='< 20 weak trend')
        axes[1].axhline(y=25, color='green', linestyle='--', label='> 25 stronger trend')

        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

        axes[1].set_title('ADXR Indicator')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Value')

        plt.tight_layout()
        plt.show()        



        
class PortOptimizer:
    def __init__(self):
        data_utils = DataUtils()
        self.AAPL = data_utils.load_historical_data("AAPL")
        self.AMZN = data_utils.load_historical_data("AMZN")
        self.GOOG = data_utils.load_historical_data("GOOG")
        self.META = data_utils.load_historical_data("META")
        self.MSFT = data_utils.load_historical_data("MSFT")
        self.NVDA = data_utils.load_historical_data("NVDA")
        self.TSLA = data_utils.load_historical_data("TSLA")

        self.AAPL_price = self.AAPL['Adj Close']
        self.AMZN_price = self.AMZN['Adj Close']
        self.GOOG_price = self.GOOG['Adj Close']
        self.META_price = self.META['Adj Close']
        self.MSFT_price = self.MSFT['Adj Close']
        self.NVDA_price = self.NVDA['Adj Close']
        self.TSLA_price = self.TSLA['Adj Close']


    def load_adjusted_adj_close_all_ticker(self):
        ticker_prices = pd.concat([self.AAPL['Adj Close'],
                            self.AMZN['Adj Close'],
                            self.GOOG['Adj Close'],
                            self.META['Adj Close'],
                            self.MSFT['Adj Close'],
                            self.NVDA['Adj Close'],
                            self.TSLA['Adj Close']], axis=1, join='inner')

        ticker_prices.columns = ['AAPL', 'AMZN', 'GOOG', 'META', 'MSFT', 'NVDA', 'TSLA']

        return ticker_prices
    

    def calculate_eReturn_covariance(self, adj_close: pd.DataFrame):
        '''
        This function calculates the covariance matrix and expected return for a given adjusted price.

        Parameters:
            adj_close (pandas.Datafrme): dataframe containing adjusted close for all tickers
        
        Returns:
            covariance_matrix(pandas.DatFrame), Expected_return(pandas.Series)
        '''

        expected_return = mean_historical_return(adj_close)
        covariance_matrix = CovarianceShrinkage(adj_close).ledoit_wolf()

        return covariance_matrix, expected_return
    

    def calculate_EfficientFrontier(self, adj_close: pd.DataFrame):
        '''
        This fuction calculates the efficient frontier and also the weights

        Parameters:
            adj_close (pandas.Datafrme): dataframe containing adjusted close for all tickers

        Returns:
            efficient_frontier (pypfopt.efficient_frontier.efficient_frontier.EfficientFrontier)
        '''

        covariance_matrix, expected_return = self.calculate_eReturn_covariance(adj_close)
        ef = EfficientFrontier(expected_return, covariance_matrix)

        return ef

        
    def clean_weights(self, adj_close: pd.DataFrame):
        '''
        This function calculates the weights and returns the clean weight for your portfolio

        Parameters:
            adj_close (pandas.Datafrme): dataframe containing adjusted close for all tickers
        
        Returns:
            clean_weight(OrderedDict)
        '''

        covariance_matrix, expected_return = self.calculate_eReturn_covariance(adj_close)
        ef = EfficientFrontier(expected_return, covariance_matrix)

        weights = ef.max_sharpe()
        clean_weight = ef.clean_weights()

        return clean_weight
    

    def plot_weights(self, adj_close: pd.DataFrame, allow_shorts=False):
        '''
        This function takes Efficient Frontier and plots the weights

        Parameters:
            adj_close (pandas.Datafrme): dataframe containing adjusted close for all tickers
            allow_shorts(bool): this will plot if we allow short selling

        Returns:
            matplotlib plot
        '''

        if allow_shorts:
            covariance_matrix, expected_return = self.calculate_eReturn_covariance(adj_close)
            ef = EfficientFrontier(expected_return, covariance_matrix, weight_bounds=(-1,1)) 
            
        else:
            covariance_matrix, expected_return = self.calculate_eReturn_covariance(adj_close)
            ef = EfficientFrontier(expected_return, covariance_matrix)   
             

        plotting.plot_weights(ef.max_sharpe())


    def plot_efficient_frontier(self, adj_close: pd.DataFrame, allow_shorts=False):
        '''
        This function takes Efficient Frontier and plots the Efficient frontier

        Parameters:
            adj_close (pandas.Datafrme): dataframe containing adjusted close for all tickers
            allow_shorts(bool): this will plot if we allow short selling

        Returns:
            matplotlib plot
        ''' 

        if allow_shorts:
            covariance_matrix, expected_return = self.calculate_eReturn_covariance(adj_close)
            ef = EfficientFrontier(expected_return, covariance_matrix)   
            ef_plot = EfficientFrontier(expected_return, covariance_matrix, weight_bounds=(-1,1))

            weights_plot = ef_plot.max_sharpe()

            ef_plot.portfolio_performance(verbose=True)

            ef_constraints = EfficientFrontier(expected_return, covariance_matrix, weight_bounds=(-1,1))

            ef_constraints.add_constraint(lambda x: cvxpy.sum(x) == 1)

            fig, ax = plt.subplots()
            ax.scatter(ef_plot.portfolio_performance()[1], ef_plot.portfolio_performance()[0], marker='*', color='r', s=200,
                    label='Tangency portfolio')

            ax.legend()
            plotting.plot_efficient_frontier(ef, ax=ax, show_assets=True)
            plt.show()

        else:
            covariance_matrix, expected_return = self.calculate_eReturn_covariance(adj_close)
            ef = EfficientFrontier(expected_return, covariance_matrix)   
            ef_plot = EfficientFrontier(expected_return, covariance_matrix, weight_bounds=(0,1))

            weights_plot = ef_plot.max_sharpe()

            ef_plot.portfolio_performance(verbose=True)

            ef_constraints = EfficientFrontier(expected_return, covariance_matrix, weight_bounds=(0,1))

            ef_constraints.add_constraint(lambda x: cvxpy.sum(x) == 1)

            fig, ax = plt.subplots()
            ax.scatter(ef_plot.portfolio_performance()[1], ef_plot.portfolio_performance()[0], marker='*', color='r', s=200,
                    label='Tangency portfolio')

            ax.legend()
            plotting.plot_efficient_frontier(ef, ax=ax, show_assets=True)
            plt.show()