
import tweepy
import pytz
import pandas as pd
from src.setting import *

def get_tweets(serch_word, min_faves, num_tweet, savedir):
    #Twitterの認証
    auth = tweepy.OAuthHandler(CK, CS)
    auth.set_access_token(AT, AS)
    api = tweepy.API(auth)
    api=tweepy.API(auth,wait_on_rate_limit=True)

    search_word = f'{serch_word} min_faves:{min_faves}'
    #何件のツイートを取得するか
    item_number = num_tweet

    #検索条件を元にツイートを抽出
    tweets = tweepy.Cursor(api.search_tweets, q=search_word, tweet_mode='extended',result_type="mixed",lang='ja').items(item_number)

    text = []
    for tweet in tweets:
        t = tweet.full_text
        if 'https://' in t:
            continue
        text.append(t)
    
    # csvでツイートを保存
    pd.Series(text).to_csv(f'{savedir}{serch_word}.csv', index=False)

    return text
