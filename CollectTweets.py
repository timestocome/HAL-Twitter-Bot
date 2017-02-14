     
# http://github.com/timestocome

# Download the current most popular tweets
# and save to file



# import standard libs
import numpy as np
import string as st 
import time 
from time import sleep
import logging 
from random import randint

              

# https://github.com/tweepy/tweepy
import tweepy
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream


# http://stackoverflow.com/questions/26965624/cant-import-requests-oauthlib
import requests
from requests_oauthlib import OAuth1Session 



# use today's data
import datetime
today = datetime.date.today()
search_date = str(today.year) + '-' + str(today.month) + '-' + str(today.day)

# use today's date in file name
collected_file_name = 'collected_tweets_' + str(today.month) + '_' + str(today.day) + '.txt'




########################################################################
# Authorization codes stored in seperate file
# so we don't accidently upload them after a late night of coding
########################################################################
from Codes import Codes
authorization_codes = Codes()

consumer_key = authorization_codes.get_consumer_key()
consumer_secret = authorization_codes.get_consumer_secret()
access_token = authorization_codes.get_access_token()
access_token_secret = authorization_codes.get_access_token_secret()







class Twitter_Api():

    def __init__(self):
    
        self._logger = logging.getLogger(__name__)

        self._consumer_key = consumer_key
        self._consumer_secret = consumer_secret

        self._access_key = access_token
        self._access_secret = access_token_secret
        
        self._authorization = None

        if consumer_key is None:

            self.tweet = lambda x : self._logger.info("Test tweet: " + x)
            self._login = lambda x : self._logger.debug("Test Login completed.")


    def _login(self):

        auth = tweepy.OAuthHandler(self._consumer_key, self._consumer_secret)
        auth.set_access_token(self._access_key, self._access_secret)

        self._authorization = auth


    
    def tweet(self, tweet):

        if self._authorization is None:
            self._login()
            pass

        api = tweepy.API(self._authorization)
        stat = api.update_status(tweet)
        self._logger.info("Tweeted: " + tweet)
        self._logger.info(stat)
    


    # will use this to collect, cleanup and store tweets
    # use most common English words as search terms 
    # the, be, to, of, and, a, in, that, have, I, it, for, not, on, with, he, as, you, do, at
    def get_popular(self):

        if self._authorization is None:
            self._login()
            pass

        api = tweepy.API(self._authorization)

        common_words = ['the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'I', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at']
        new_tweets = []

        for w in common_words:
            print("____________ w ", w)
            tweets = tweepy.Cursor(api.search, 
                                    q=w, 
                                    since=search_date,
                                    result_type='popular').items()
            for t in tweets:
                tweet = t.text
                new_tweets.append(tweet.encode('utf-8'))    

            with open(collected_file_name, 'a') as myfile:
                for t in new_tweets:
                    myfile.write(str(t)+"\n")
            sleep(60)


    def disconnect(self):
        self._authorization = None




    



####################################################################################
# run code
####################################################################################
# fetch stored authorization codes
twitter_api = Twitter_Api()

# get today's popular tweets
twitter_api.get_popular()


# clean up the tweets
from Cleanup_collected_tweets import cleanup_tweets
cleanup_tweets()


# create word cloud
from WordCloud import generate_current_wordcloud
generate_current_wordcloud()



'''

#############################
# automate uploading image
# http://stackoverflow.com/questions/31748444/how-to-update-twitter-status-with-image-using-image-url-in-tweepy
'''
def tweet_image(url, message):
    api = twitter_api()
    filename = 'temp.jpg'
    request = requests.get(url, stream=True)
    if request.status_code == 200:
        with open(filename, 'wb') as image:
            for chunk in request:
                image.write(chunk)

        api.update_with_media(filename, status=message)
        os.remove(filename)

