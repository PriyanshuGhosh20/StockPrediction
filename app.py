import streamlit as st
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

# Set the page title and description
st.set_page_config(page_title="Financial News Sentiment Analysis", page_icon=":money_with_wings:")

# Set up the page layout
st.title("Financial News Sentiment Analysis")
st.write("This app analyzes the sentiment of financial news headlines to make a stock market prediction.")
url = st.text_input("Enter the URL of a financial news website:")
if st.button("Analyze"):
    # Send a GET request to the URL
    response = requests.get(url)

    # Parse the HTML content of the page using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all the news headlines on the page
    headlines = soup.find_all(['h1','h2','h3','h4','h5'])
    headlines = headlines[:50]

    # Create a list to store the sentiment scores of each headline
    sentiment_scores = []

    # Create a list to store the tickers for each headline
    tickers = []

    # Initialize the Vader sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()

    # Loop through each headline and perform sentiment analysis using Vader
    for headline in headlines:
        text = headline.text
        ticker = text[:10]
        sentiment_score = analyzer.polarity_scores(text)['compound']
        if sentiment_score:
            tickers.append(ticker)
            sentiment_scores.append(sentiment_score)
    # Check if the sentiment_scores list is empty before calculating the average score
    if sentiment_scores:
        # Calculate the average sentiment score for all the headlines
        average_score = sum(sentiment_scores) / len(sentiment_scores)

        # Use the sentiment score to make a stock market prediction
        if average_score > 0:
            st.write("The stock market is likely to go up.")
        elif average_score < 0:
            st.write("The stock market is likely to go down.")
        else:
            st.write("The stock market is likely to remain stable.")

        # Plot the sentiment scores on a bar chart
        cmap = plt.cm.get_cmap('RdYlGn')
        normalize = plt.Normalize(vmin=-1, vmax=1)
        colors = [cmap(normalize(score)) for score in sentiment_scores]
        fig, ax = plt.subplots(figsize=(25,10))
        ax.bar(tickers, sentiment_scores, width=0.5, color=colors)
        ax.set_xticklabels(tickers, rotation=90)
        ax.set_xlabel('Tickers')
        ax.set_ylabel('Sentiment Scores')
        st.pyplot(fig)

    else:
        st.write("No headlines found on the page.")
