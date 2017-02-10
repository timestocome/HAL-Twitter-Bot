

# http://github.com/timestocome
# https://apps.twitter.com/app/13350654


# starter code 
# https://www.pygaze.org/2016/03/how-to-code-twitter-bot/
# https://github.com/esdalmaijer


############################################################################
# setup
#############################################################################
# import standard libs
import numpy as np
import string as st 
import time 
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



#######################################################################
# read in text and generate tweets from stored text 
#######################################################################
from Markov_tweet_generator import Markov_tweet_generator

generated_tweets = Markov_tweet_generator()
n_tweets = len(generated_tweets)

print(n_tweets)
print(generated_tweets)




##############################################################################
# bot code to push created tweets to twitter stream
##############################################################################
class Bot:

    def __init__(self, twitter_api, tweets_per_hour = 60):

        self._twitter_api = twitter_api
        self._logger = logging.getLogger(__name__)

        #Calculate sleep timer
        self.sleep_timer = int(60*60/tweets_per_hour)

   
    def _get_tweet(self):
        tweet = ''
        
        tweet = sentences[randint(0, len(sentences)-1)]
        #print("tweet", tweet)

        return tweet



    def push_tweet(self):

        while True:
            try:
                tweet = self._get_tweet()
                self._twitter_api.tweet(tweet)

            except Exception as e:
                self._logger.error(e, exc_info=True)
                self._twitter_api.disconnect()

            time.sleep(self.sleep_timer) #Every 10 minutes


   
  
        


#######################################################################
# set up Twitter api and authorizations
#####################################################################
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
    # markov chain will then be built from these instead of Alice
    def get_popular(self):
        if self._authorization is None:
            self._login()
            pass

        api = tweepy.API(self._authorization)

        new_tweets = []
        tweets = tweepy.Cursor(api.search, q="saturday", result_type='popular').items()
        for t in tweets:
            tweet = t.text
            print(tweet.encode('utf-8'))
            new_tweets.append(tweet.encode('utf-8'))

        with open('collected_tweets.txt', 'a') as myfile:
            for t in new_tweets:
                myfile.write(str(t)+"\n")
                


    def disconnect(self):
        self._authorization = None




    



####################################################################################
# run code
####################################################################################
twitter_api = Twitter_Api()

# create and post tweets
#bot = Bot(twitter_api)
#bot.push_tweet()
