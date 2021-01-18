# ======================================================================================================================
#                                                   Imports
# ======================================================================================================================

import numpy as np
import matplotlib.pyplot as plt
import re
from twython import Twython
from PIL import Image
from wordcloud import WordCloud, STOPWORDS
from IPython.display import Image as im
import pickle
from os import path, mkdir
from datetime import datetime
import pandas as pd
from mysql import connector
from textblob import TextBlob
import plotly.express as px
import nltk
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')

# import gensim
# from gensim.models import Word2Vec

# ======================================================================================================================
#                                                   Constants
# ======================================================================================================================

current_time = datetime.now().strftime("%d_%m__%H_%M_%S")
GENERAL_OUTPUT_DIR = <YOUR_OUTPUT_FOLDER_PATH>
OUTPUT_DIR = path.join(GENERAL_OUTPUT_DIR, current_time)
mkdir(OUTPUT_DIR)

BRAND_NAME = 'Tesla'
N = 1000
LOGO_PATH = 'tesla_logo.jpg'

#Connect to Twitter
APP_KEY = <YOUR TWITTER APP KEY>
APP_SECRET = <YOUR TWITTER APP SECRET>

KEYS = ['created_at', 'text', "id_str"]

# ======================================================================================================================
#                                                   Main
# ======================================================================================================================


def get_data_from_db():
    db_connection = connector.connect(
        host="host",
        user="root",
        passwd="password",
        database="TwitterDB",
        charset = 'utf8'
     )
    time_now = datetime.datetime.utcnow()
    time_10mins_before = datetime.timedelta(hours=0,minutes=10).strftime('%Y-%m-%d %H:%M:%S')
    time_interval = time_now - time_10mins_before
    query = "SELECT id_str, text, created_at, polarity,          \
             user_location FROM {} WHERE created_at >= '{}'"     \
            .format(settings.TABLE_NAME, time_interval)
    df = pd.read_sql(query, con=db_connection)
    df.to_csv(path.join(OUTPUT_DIR, 'db_twits.csv'))



def clean_tweet(raw_string):
    # Create a string form of our list of text
    no_links = re.sub(r'http\S+', '', raw_string)
    no_unicode = re.sub(r"\\[a-z][a-z]?[0-9]+", '', no_links)
    no_special_characters = re.sub('[^A-Za-z ]+', '', no_unicode)

    # clean words
    words = no_special_characters.split(" ")
    words = [w for w in words if len(w) > 2]  # ignore a, an, be, ...
    words = [w.lower() for w in words]
    words = [w for w in words if w not in STOPWORDS]
    return words

def get_df_from_query():
    # Extract textfields from tweets
    twitter = Twython(APP_KEY, APP_SECRET)
    search_results = twitter.search(count=N, q=BRAND_NAME)
    with open(path.join(OUTPUT_DIR, 'search_results.pkl'), "wb") as f:
        pickle.dump(search_results, f)
    raw_tweets = []
    clean_tweets = []
    df_data = []
    for tweet in search_results['statuses']:
        if tweet['metadata']['iso_language_code'] !='en':
            continue
        raw_tweets.append(tweet['text'])
        tweet_data = {k:tweet[k] for k in KEYS}
        tweet_data['location'] = tweet.get('user', {}).get('location')

        clean_text = clean_tweet(tweet['text'])
        clean_tweets.append(clean_text)

        sentiment = TextBlob(' '.join(clean_text)).sentiment
        polarity = sentiment.polarity
        subjectivity = sentiment.subjectivity
        tweet_data.update({
            'polarity': polarity,
            'subjectivity': subjectivity
        })
        df_data.append(tweet_data)

    df = pd.DataFrame(df_data)
    with open(path.join(OUTPUT_DIR, 'raw_tweets.pkl'), "wb") as f:
        pickle.dump(raw_tweets, f)
    with open(path.join(OUTPUT_DIR, 'clean_tweets.pkl'), "wb") as f:
        pickle.dump(clean_tweets, f)
    df.to_csv(path.join(OUTPUT_DIR, 'df.csv'))
    return df, clean_tweets

def create_gui(df):
    # UTC for date time at default
    # df = get_data_from_db()
    df['created_at'] = pd.to_datetime(df['created_at'])

    # Clean and transform data to enable time series
    result = df.groupby([pd.Grouper(key='created_at', freq='5s'), 'polarity']).count().unstack(fill_value=0).stack().reset_index()
    result = result.rename(columns={
        "id_str": "Num of '{}' mentions".format(BRAND_NAME),
        "created_at": "Time in UTC"
    })

    time_series = result["Time in UTC"][result['polarity'] == 0].reset_index(drop=True)
    fig = px.line(result, x='Time in UTC', y="Num of '{}' mentions".format(BRAND_NAME), color='polarity')
    fig.show()

    content = ' '.join(df["text"])
    content = re.sub(r"http\S+", "", content)
    content = content.replace('RT ', ' ').replace('&amp;', 'and')
    content = re.sub('[^A-Za-z0-9]+', ' ', content)
    content = content.lower()

    tokenized_word = word_tokenize(content)
    stop_words = set(stopwords.words("english"))
    filtered_sent = []
    for w in tokenized_word:
        if w not in stop_words:
            filtered_sent.append(w)
    fdist = FreqDist(filtered_sent)
    fd = pd.DataFrame(fdist.most_common(20), columns=["Word", "Frequency"]).drop([0]).reindex()
    fig = px.bar(fd, x="Word", y="Frequency")
    fig.update_traces(marker_color='rgb(240,128,128)',
                      marker_line_color='rgb(8,48,107)',
                      marker_line_width=1.5, opacity=0.8)
    fig.show()

def words_similarity(clean_tweets):
    # Create Skip Gram model
    model = gensim.models.Word2Vec(clean_tweets, min_count=1, size=100,
                                    window=5, sg=1)
    WORDS = ['money', 'expensive', 'cheap', 'affordable', 'vehicle', 'musk', 'car']
    # Print results
    for word in WORDS:
        dist = model.similarity(word, BRAND_NAME)
        print("Cosine similarity between '{}' and '{}' (Skip Gram model) : {}".format(word, BRAND_NAME, dist))


def hist_polarity(df):
    values = df['polarity'].to_numpy()
    weights = np.ones_like(values) / float(len(values))
    plt.hist(values, bins=30, range=[-1,1], weights=weights)
    plt.grid()
    plt.title('Tesla related tweets sentiment polarity (N={})'.format(N))
    plt.savefig(path.join(OUTPUT_DIR, 'polarity3.png'))
    plt.xlabel('sentiment polarity value')
    plt.ylabel('Normalised frequencies')
    plt.xlim([-1,1])
    plt.show()


def create_cloud_of_words(tweets):
    # create word cloud
    mask = np.array(Image.open(LOGO_PATH))
    wc = WordCloud(background_color="white", max_words=200, mask=mask)
    all_words = []
    for t in tweets:
        all_words += t
    clean_string = ','.join(all_words)
    wc.generate(clean_string)

    # show
    f = plt.figure(figsize=(40, 40))
    f.add_subplot(1, 1, 1)
    plt.imshow(wc, interpolation='bilinear')
    plt.title('Twitter cloud of words for Tesla', size=40)
    plt.axis("off")
    plt.savefig(path.join(OUTPUT_DIR, 'cloud_of_words_tesla.png'))
    plt.show()


if __name__ == '__main__':
    df, clean_tweets = get_df_from_query()
    # words_similarity(clean_tweets)
    create_gui(df)
    hist_polarity(df)
    create_cloud_of_words(clean_tweets)



