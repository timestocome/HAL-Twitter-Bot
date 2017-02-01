
# starter code 
# https://gist.github.com/yanofsky/5436496



# import other libs 
# https://github.com/jsvine/markovify
import markovify                    

# https://github.com/tweepy/tweepy
import tweepy
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream


# http://stackoverflow.com/questions/26965624/cant-import-requests-oauthlib
import requests
from requests_oauthlib import OAuth1Session 

import csv


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
# scrape Tweets

def get_all_tweets(screen_name):
	#Twitter only allows access to a users most recent 3240 tweets with this method
	
	#authorize twitter, initialize tweepy
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)
	
	#initialize a list to hold all the tweepy Tweets
    alltweets = []	
	
	#make initial request for most recent tweets (200 is the maximum allowed count)
    new_tweets = api.user_timeline(screen_name = screen_name,count=200)
	
	#save most recent tweets
    alltweets.extend(new_tweets)
	
	#save the id of the oldest tweet less one
    oldest = alltweets[-1].id - 1
	
	#keep grabbing tweets until there are no tweets left to grab
    while len(new_tweets) > 0:
        
        print("getting tweets before %s" % (oldest))
		
		#all subsiquent requests use the max_id param to prevent duplicates
        new_tweets = api.user_timeline(screen_name = screen_name,count=200,max_id=oldest)
		
		#save most recent tweets
        alltweets.extend(new_tweets)
		
		#update the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1
		
        print ("...%s tweets downloaded so far" % (len(alltweets)))
	
	#transform the tweepy tweets into a 2D array that will populate the csv	
    #outtweets = [[tweet.id_str, tweet.created_at, tweet.text.encode("utf-8")] for tweet in alltweets]
	        
    tweets = [[tweet.text.encode("utf-8")] for tweet in alltweets]

    with open('breitbart_tweets.txt', 'a') as myfile:
            for t in tweets:
                myfile.write(str(t)+"\n")



    
#get_all_tweets("BuzzFeed")
#get_all_tweets("Upworthy")
get_all_tweets("BreitbartNews")