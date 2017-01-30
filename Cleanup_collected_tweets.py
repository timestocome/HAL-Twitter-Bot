


############################################################################
# setup
#############################################################################
# import standard libs
import numpy as np
import string as st 





#######################################################################
# read in text and clean it up
#######################################################################
tweets_data = []
with open('collected_tweets.txt', encoding='utf-8') as f:
    for line in f:
        tweets_data.append(line)
    


print(len(tweets_data))

for t in tweets_data:
    print(t[2:-2])