import matplotlib.pyplot as plt
import nltk
import pandas as pd
import regex as re
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud


class LanguageProcessing:
    def __init__(self):
        self.download_nltk_corpus()
        self.stop_words = set(stopwords.words("english"))
        self.punctuation_removal = re.compile(r"[^\w\s]")
        self.sia = SentimentIntensityAnalyzer()

    def download_nltk_corpus(self):

        try:
            nltk.download("stopwords", quiet=True)
            nltk.download("punkt", quiet=True)
            nltk.download("wordnet", quiet=True)
            nltk.download("vader_lexicon", quiet=True)
        except Exception as e:
            print(f"An error occurred while downloading NLTK corpora: {e}")

    def get_lemma(self, word):
        lemma = wn.morphy(word)
        if lemma is None:
            return word
        else:
            return lemma

    def clean_data(self, text):

        tokens = word_tokenize(text.lower())
        stripped = [
            self.punctuation_removal.sub("", token) for token in tokens if token
        ]
        words = [
            self.get_lemma(word)
            for word in stripped
            if word.isalpha() and word not in self.stop_words
        ]

        return " ".join(words)

    def sentiment_analysis(self, headline):

        sentiment_score = self.sia.polarity_scores(headline)["compound"]

        return sentiment_score

    def categorize_headline(self, sentiment_score):

        if sentiment_score >= 0.90:
            return "Extremely Positive"
        elif sentiment_score <= -0.90:
            return "Extremely Negative"
        elif sentiment_score > 0.5:
            return "Positive"
        elif sentiment_score < -0.5:
            return "Negative"
        else:
            return "Neutral"

    def TopicModel_TFIDF(self, data: pd.DataFrame):
        vectorizer = TfidfVectorizer(max_df=0.2, min_df=0.01, stop_words="english")

        train_dtm = vectorizer.fit_transform(data.headline)

        train_token_count = train_dtm.sum(0).A.squeeze()
        tokens = vectorizer.get_feature_names_out()
        word_count = pd.Series(train_token_count, index=tokens).sort_values(
            ascending=False
        )

        return word_count

    def plot_wordCloud(self, word_count):
        wordcloud = WordCloud(
            width=800, height=400, background_color="white"
        ).generate_from_frequencies(word_count)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")  # No axes for the word cloud
        plt.show()
