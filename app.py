import streamlit as st
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import validators

# Download the VADER lexicon for sentiment analysis
nltk.download('vader_lexicon')

# Set the title and icon for the app
st.set_page_config(page_title="Stock Market Sentiment Analysis", page_icon=":money_with_wings:")

# Display the main title
st.title("Stock Market Sentiment Analysis")

# Ask the user for a financial news website URL to analyze
url = st.text_input("Enter the URL of a financial news website:", placeholder="E.g. https://www.businesstoday.in")

# Allow the user to select the number of headlines to analyze
num_headlines = st.slider("Select the number of headlines to analyze:", min_value=1, max_value=100, value=10)

# Allow the user to select the type of visualization to display
visualization_type = st.selectbox("Select the type of visualization to display:",options=["Bar Chart", "Line Chart","Area Chart","Scatter Chart"])

# Analyze the news headlines when the user clicks the "Analyze" button
try:
    if st.button("Analyze"):
        # Validate the URL
        if not validators.url(url):
            st.error("Please enter a valid URL.")
        else:
            # Retrieve the webpage HTML
            with st.spinner('Analyzing the news headlines...'):
                response = requests.get(url)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                # Find the news headlines and their ticker symbols
                headlines = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5'])[:num_headlines]
                sentiment_scores = []
                tickers = []
                analyzer = SentimentIntensityAnalyzer()
                # Analyze the sentiment of each headline and save the scores and ticker symbols
                for i, headline in enumerate(headlines):
                    text = headline.text
                    ticker = text[:10]
                    sentiment_score = analyzer.polarity_scores(text)['compound']
                    if sentiment_score:
                        tickers.append(ticker)
                        sentiment_scores.append(sentiment_score)
                # Display the sentiment prediction based on the average score
                if sentiment_scores:
                    average_score = sum(sentiment_scores) / len(sentiment_scores)

                    if average_score > 0:
                        st.markdown(f"<h2 style='color: green;'>The stock market is likely to go up.</h2>",
                                    unsafe_allow_html=True)
                    elif average_score < 0:
                        st.markdown(f"<h2 style='color: red;'>The stock market is likely to go down.</h2>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<h2 style='color: yellow;'>The stock market is likely to remain stable.</h2>", unsafe_allow_html=True)

                    # Display the selected type of visualization
                    if visualization_type == "Bar Chart":
                        # set color map for the plot
                        cmap = plt.cm.get_cmap('RdYlGn')
                        # normalize the data and set color values
                        normalize = plt.Normalize(vmin=-1, vmax=1)
                        colors = [cmap(normalize(score)) for score in sentiment_scores]
                        # create the plot
                        fig, ax = plt.subplots(figsize=(25, 10))
                        ax.bar(tickers, sentiment_scores, width=0.5, color=colors)
                        ax.set_xticklabels(tickers, rotation=90)
                        ax.set_xlabel('Tickers')
                        ax.set_ylabel('Sentiment Scores')
                        # show the plot
                        st.pyplot(fig)

                    elif visualization_type == "Line Chart":
                        # sort the data by ticker
                        sorted_data = sorted(zip(tickers, sentiment_scores), key=lambda x: x[0])
                        sorted_tickers = [data[0] for data in sorted_data]
                        sorted_scores = [data[1] for data in sorted_data]  
                        # create the plot
                        fig, ax = plt.subplots(figsize=(25, 10))
                        ax.plot(sorted_tickers, sorted_scores, marker='o')
                        ax.set_xticklabels(sorted_tickers, rotation=90)
                        ax.set_xlabel('Tickers')
                        ax.set_ylabel('Sentiment Scores')
                        # show the plot
                        st.pyplot(fig)

                    elif visualization_type == "Area Chart":
                        # set color map for the plot
                        cmap = plt.cm.get_cmap('RdYlGn')
                        # normalize the data and set color values
                        normalize = plt.Normalize(vmin=-1, vmax=1)
                        colors = [cmap(normalize(score)) for score in sentiment_scores]
                        # create the plot
                        fig, ax = plt.subplots(figsize=(25, 10))
                        ax.fill_between(tickers, sentiment_scores, 0, color=colors)
                        ax.set_xticklabels(tickers, rotation=90)
                        ax.set_xlabel('Tickers')
                        ax.set_ylabel('Sentiment Scores')
                        # show the plot
                        st.pyplot(fig)

                    elif visualization_type == "Scatter Chart":
                        # set color map for the plot
                        cmap = plt.cm.get_cmap('RdYlGn')
                        # normalize the data and set color values
                        normalize = plt.Normalize(vmin=-1, vmax=1)
                        colors = [cmap(normalize(score)) for score in sentiment_scores]
                        # create the plot
                        fig, ax = plt.subplots(figsize=(25, 10))
                        ax.scatter(tickers, sentiment_scores, color=colors,s=200)
                        ax.set_xticklabels(tickers, rotation=90)
                        ax.set_xlabel('Tickers')
                        ax.set_ylabel('Sentiment Scores')
                        # show the plot
                        st.pyplot(fig)
                else:
                    st.write("No headlines found on the page.")
except Exception as e:
    st.error(f"An error occurred while processing the request: {str(e)}")
