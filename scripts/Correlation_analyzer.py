import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import seaborn as sns


import os 
import sys

sys.path.append(os.path.abspath("../scripts"))

from utils import *
from language_processing import *
from financial_analyzer import *

class PrepData:
    def __init__(self, ticker: str):

        self.data_utils = DataUtils()
        self.LangProcess = LanguageProcessing()

        self.news_data = self.data_utils.load_news_data(ticker)
        self.price_data = self.data_utils.load_historical_data(ticker)
        

    def _sentiment_analysis(self):

        self.news_data["cleaned_headline"] = self.news_data["headline"].apply(
                lambda word: self.LangProcess.clean_data(word)
        )

        self.news_data.dropna(inplace=True)

        self.news_data["sentiment"] = self.news_data["cleaned_headline"].apply(
            lambda word: self.LangProcess.sentiment_analysis(word)
        )

        self.news_data.index = pd.to_datetime(self.news_data.index)
        self.news_data.index = self.news_data.index.strftime('%Y-%m-%d')
        


    def _merge_price_news(self):

        sentiment = self.news_data['sentiment']

        sentiment = sentiment.groupby(by=sentiment.index).mean()
        sentiment.index = pd.to_datetime(sentiment.index)
        
        price_sentiment = self.price_data.merge(sentiment,  right_on=sentiment.index, left_on=self.price_data.index, sort=True )
        price_sentiment = price_sentiment.rename({'key_0': 'Date'}, axis=1).set_index('Date')

        price_sentiment['Daily returns'] = price_sentiment['Adj Close'].pct_change()
        price_sentiment['Pct sentiment'] = price_sentiment['sentiment'].pct_change()
        

        return price_sentiment
    

class CorrAnalysis(PrepData):
    def __init__(self, ticker):
        super().__init__(ticker)
        self._sentiment_analysis()
        self.price_sentiment = self._merge_price_news()

    def _plot_relations(self, cols: list, price_sentiment):
        
        cols = [col for col in cols if col != 'sentiment' and col != 'Pct sentiment']
        for col in cols:
            fig, ax1 = plt.subplots(nrows= 2, figsize=(12, 10))

            ax1[0].plot(price_sentiment.index, price_sentiment.sentiment, color='b', label='sentiment')
            ax1[1].plot(price_sentiment.index, price_sentiment['Pct sentiment'], color='b', label='percent change in sentiment')
            ax1[0].set_xlabel('Date')
            ax1[0].set_ylabel('Sentiment', color='b')
            ax1[1].set_ylabel('Percent Sentiment', color='b')
            ax1[0].tick_params(axis='y', labelcolor='b')
            ax1[1].tick_params(axis='x', rotation=30)
            ax1[0].tick_params(axis='x', rotation=30)

            # Create a secondary y-axis for the columns
            ax2 = ax1[0].twinx()
            ax3 = ax1[1].twinx()
            ax2.plot(price_sentiment.index, price_sentiment[col], color='r', label=f'{col}')
            ax3.plot(price_sentiment.index, price_sentiment[col], color='r', label=f'{col}')
            ax2.set_ylabel(f'{col}', color='r')
            ax3.set_ylabel(f'{col}', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            ax3.tick_params(axis='x', rotation=30)
            ax2.tick_params(axis='x', rotation=30)

            ax1[0].set_title(f'Sentiment vs {col}')
            ax1[1].set_title(f'Percent Sentiment vs {col}')


            plt.subplots_adjust(hspace=0.4)
            plt.show()

    def correlate_indicator_returns(self, indicator: str):
        
        analyze_price_sentiment = self.price_sentiment
        technical_indicator = TechnicalIndicator(analyze_price_sentiment)
        analyze_price_sentiment.dropna(inplace=True)

        if indicator == 'leading':
            technical_indicator.leading_indicators()
            cols = ['Daily returns', 'TSF', 'CMO', 'sentiment', 'Pct sentiment', 'Close']
        elif indicator == 'lagging':
            technical_indicator.lagging_indicators()
            cols = ['Daily returns', 'sentiment', 'Pct sentiment', 'Close', 'macd','RSI', 'ADXR','WMA', 'EMA' ]
        elif indicator == 'instantaneous':
            technical_indicator.instantaneous_indicators()
            cols = ['Daily returns', 'sentiment', 'Pct sentiment', 'Close','TYPPRICE' ]
        elif indicator == 'volume':
            technical_indicator.volume_indicators()
            cols = ['Daily returns', 'sentiment', 'Pct sentiment', 'Close', 'MFI', 'OBV', 'AD']
        elif indicator == 'volatility':
            technical_indicator.volatility_indicators()
            cols = ['Daily returns', 'sentiment', 'Pct sentiment', 'Close', 'ATR', 'NATR']
            
        corr_matrix = analyze_price_sentiment[cols].corr().to_numpy()

        
        # Maptplotlib heatmap
        fig, ax = plt.subplots(figsize=(8, 6))  
        cax = ax.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest')
        plt.colorbar(cax, ax=ax, orientation='vertical')
        ax.set_xticks(np.arange(len(cols)))
        ax.set_yticks(np.arange(len(cols)))
        ax.set_xticklabels(cols)
        ax.set_yticklabels(cols)
        plt.xticks(rotation=45, ha='right')
        # Add annotations
        for i in range(corr_matrix.shape[0]):
            for j in range(corr_matrix.shape[1]):
                text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}', ha='center', va='center', color='black')

        ax.set_title('Correlation analysis') 
        plt.tight_layout()
        plt.show()

        # Better way use plotly plot
        # fig = px.imshow(corr_matirx, text_auto=True)
        # fig.update_layout(
        #     plot_bgcolor='rgba(0,0,0,0)',  
        #     paper_bgcolor='rgba(0,0,0,0)',
        #     xaxis= dict(color='white'),
        #     yaxis= dict(color='white'),
        # )
        # fig.show()



        self._plot_relations(cols, analyze_price_sentiment)


