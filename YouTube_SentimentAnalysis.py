import pickle
from googleapiclient.discovery import build # Used to search in YT & get the url
import argparse
import numpy as np
import unidecode
import time
import os
import pandas as pd
import tweepy as tw
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from textblob import TextBlob  
import streamlit as st
import matplotlib.pyplot as plt

DEVELOPER_KEY = "AIzaSyCI_dU6TZ_XBKSjeovcZrWMgIHW7FuuclY"
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"
videoIds = []
# st.set_option('deprecation.showPyplotGlobalUse', False)

def sentiment(polarity):
    if polarity < 0:
        p = "Negative"
    elif polarity > 0:
        p = "Positive"
    else:
        p = "Neutral"
    return p

model=pickle.load(open('model.pkl','rb'))
def clasifier(text):
    input=np.array([[text]]).astype(np.float64)
    prediction=model.predict(input)
    pred='{0:.{1}f}'.format(prediction[0][0], 2)
    com = comments(list)
    com_list = [i.text for i in com]
    p = [i for i in clasifier(com_list)]
    q=[p[i]['label'] for i in range(len(p))]
    r=[p[i]['score'] for i in range(len(p))]
    return (pred) 

def scrape_comments(youtube_video_url, Movie_name):
    # st.title("Youtube")
    chrome_path = Service("D:\chromedriver.exe")

    driver = webdriver.Chrome(service=chrome_path)
    driver.get(youtube_video_url)
    driver.maximize_window()
    driver.implicitly_wait(30)

    # Scrolling to load comments
    time.sleep(20)
    driver.execute_script('window.scrollTo(0,750);')
    time.sleep(10)
    print("scrolled")
    sort = driver.find_element(by=By.XPATH, value=
        """//*[@id="icon-label"]""")  # Sorting by top comments
    sort.click()
    time.sleep(10)
    topcomments = driver.find_element(by=By.XPATH, value=
        """//*[@id="menu"]/a[1]/tp-yt-paper-item/tp-yt-paper-item-body/div[1]""")
    topcomments.click()
    time.sleep(20)
    for i in range(0, 2):
        driver.execute_script(
            "window.scrollTo(0,Math.max(document.documentElement.scrollHeight,document.body.scrollHeight,document.documentElement.clientHeight))")
        time.sleep(10)
    totalcomments = len(driver.find_elements(by=By.XPATH, value=
        """//*[@id="content-text"]"""))
    if totalcomments < 40:
        index = totalcomments
    else:
        index = 40
    ccount = 0
    comments = []
    while ccount < index:
        try:
            comment = driver.find_elements(by=By.XPATH, value=
                '//*[@id="content-text"]')[ccount].text
            ccount = ccount + 1
            print(comment)
            comments.append(comment)
        except:
            comment = ""
    polarity = []
    subjectivity = []
    sentiment_type = []

    for elm in comments:
        x = TextBlob(elm)
        print(elm)
        print("Polarity : " + str(x.sentiment.polarity))
        print("Subjectivity : " + str(x.sentiment.subjectivity))
        polarity.append(x.sentiment.polarity)
        subjectivity.append(x.sentiment.subjectivity)
        s = sentiment(x.sentiment.polarity)
        print("Sentiment Type :" + s)
        sentiment_type.append(s)

    dataframe = {"comment": comments, "sentiment_type": sentiment_type, "polarity": polarity,
                 "subjectivity": subjectivity}
    df = pd.DataFrame.from_dict(dataframe, orient='index')
    df1 = df.transpose()
    df1.columns = ['comment', 'polarity', 'sentiment_type', 'subjectivity']
    df1.to_csv(r"E:/FrAgnel/Sem 6/Sentiment Analysis/comment_sentiment_" + Movie_name + ".csv", header=True,
               encoding='utf-8', index=False)
    


# def viualizaiton():
#     title_type = df.groupby('sentiment').agg('count')
#     print(title_type)
#     piechart=df.sentiment.value_counts().plot(kind='pie',autopct="%1.0f%%")
#     st.write(piechart)
#     st.pyplot()

def youtube_video_url(options):
    youtube = build(YOUTUBE_API_SERVICE_NAME,
                    YOUTUBE_API_VERSION, developerKey=DEVELOPER_KEY)
    search_response = youtube.search().list(q=options.q, part="id,snippet",
                                            maxResults=options.max_results).execute()
    for search_result in search_response.get("items", []):
        if search_result["id"]["kind"] == "youtube#video":
            videoId = search_result["id"]["videoId"]
            print("videoId: " + str(videoId))
            videoIds.append(videoId)
            url = "https://www.youtube.com/watch?v="+videoId
            print(url)
    return url


if __name__ == "__main__":
    print("Enter the video name: ")
    Movie_name = str(input())
    parser = argparse.ArgumentParser(description='youtube search')
    parser.add_argument("--q", help="Search term",
                        default=Movie_name)
    parser.add_argument("--max-results", help="Max results", default=1)
    args = parser.parse_args()
    youtube_video_url = youtube_video_url(args)
    scrape_comments(youtube_video_url, Movie_name)
