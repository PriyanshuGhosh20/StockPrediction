import streamlit as st
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import validators

nltk.download('vader_lexicon')

st.set_page_config(page_title="Stock Market Sentiment Analysis", page_icon=":money_with_wings:")

st.title("Stock Market Sentiment Analysis")

st.write("This app analyzes the sentiment of financial news headlines to make a stock market prediction.")

url = st.text_input("Enter the URL of a financial news website:", placeholder="E.g. https://www.businesstoday.in")

num_headlines = st.slider("Select the number of headlines to analyze:", min_value=1, max_value=100, value=10)

visualization_type = st.selectbox("Select the type of visualization to display:",options=["Bar Chart", "Line Chart","Area Chart","Scatter Chart"])

try:
    if st.button("Analyze"):
        if not validators.url(url):
            st.error("Please enter a valid URL.")
        else:
            with st.spinner('Analyzing the news headlines...'):
                response = requests.get(url)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                headlines = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5'])[:num_headlines]
                sentiment_scores = []
                tickers = []
                analyzer = SentimentIntensityAnalyzer()

                for i, headline in enumerate(headlines):
                    text = headline.text
                    ticker = text[:10]
                    sentiment_score = analyzer.polarity_scores(text)['compound']
                    if sentiment_score:
                        tickers.append(ticker)
                        sentiment_scores.append(sentiment_score)
                if sentiment_scores:
                    average_score = sum(sentiment_scores) / len(sentiment_scores)

                    if average_score > 0:
                        st.markdown(f"<h2 style='color: green;'>The stock market is likely to go up.</h2>",
                                    unsafe_allow_html=True)
                    elif average_score < 0:
                        st.markdown(f"<h2 style='color: red;'>The stock market is likely to go down.</h2>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<h2 style='color: yellow;'>The stock market is likely to remain stable.</h2>", unsafe_allow_html=True)

                    if visualization_type == "Bar Chart":
                        cmap = plt.cm.get_cmap('RdYlGn')
                        normalize = plt.Normalize(vmin=-1, vmax=1)
                        colors = [cmap(normalize(score)) for score in sentiment_scores]
                        fig, ax = plt.subplots(figsize=(25, 10))
                        ax.bar(tickers, sentiment_scores, width=0.5, color=colors)
                        ax.set_xticklabels(tickers, rotation=90)
                        ax.set_xlabel('Tickers')
                        ax.set_ylabel('Sentiment Scores')
                        st.pyplot(fig)

                    elif visualization_type == "Line Chart":
                        sorted_data = sorted(zip(tickers, sentiment_scores), key=lambda x: x[0])
                        sorted_tickers = [data[0] for data in sorted_data]
                        sorted_scores = [data[1] for data in sorted_data]   
                        fig, ax = plt.subplots(figsize=(25, 10))
                        ax.plot(sorted_tickers, sorted_scores, marker='o')
                        ax.set_xticklabels(sorted_tickers, rotation=90)
                        ax.set_xlabel('Tickers')
                        ax.set_ylabel('Sentiment Scores')
                        st.pyplot(fig)

                    elif visualization_type == "Area Chart":
                        cmap = plt.cm.get_cmap('RdYlGn')
                        normalize = plt.Normalize(vmin=-1, vmax=1)
                        colors = [cmap(normalize(score)) for score in   sentiment_scores]
                        fig, ax = plt.subplots(figsize=(25, 10))
                        ax.fill_between(tickers, sentiment_scores, 0, color=colors)
                        ax.set_xticklabels(tickers, rotation=90)
                        ax.set_xlabel('Tickers')
                        ax.set_ylabel('Sentiment Scores')
                        st.pyplot(fig)

                    elif visualization_type == "Scatter Chart":
                        cmap = plt.cm.get_cmap('RdYlGn')
                        normalize = plt.Normalize(vmin=-1, vmax=1)
                        colors = [cmap(normalize(score)) for score in sentiment_scores]
                        fig, ax = plt.subplots(figsize=(25, 10))
                        ax.scatter(tickers, sentiment_scores, color=colors,s=200)
                        ax.set_xticklabels(tickers, rotation=90)
                        ax.set_xlabel('Tickers')
                        ax.set_ylabel('Sentiment Scores')
                        st.pyplot(fig)
                else:
                    st.write("No headlines found on the page.")
        
except Exception as e:
    st.error(f"An error occurred while processing the request: {str(e)}")
