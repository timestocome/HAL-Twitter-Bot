# http://github.com/timestocome

# read in tweets that were written to a disk file and clean them up
# remove unicode, user names, links, hashtags


############################################################################
# setup
#############################################################################
import numpy as np
import string as st 

from itertools import chain
import re



#######################################################################
# read in saved tweets and clean them up
#######################################################################
tweets_data = []
with open('collected_tweets.txt', encoding='utf-8') as f:
    for line in f:
        tweets_data.append(line)
    


# remove 'b...'
cleaned_tweets = []
for t in tweets_data:
    cleaned_tweet = (t[2:-2])
    cleaned_tweets.append(cleaned_tweet)


# remove @userName
# remove links
# remove hashtags
cleaned_data = []
for text in cleaned_tweets:
    new_text = re.sub(r'#\w+ ?', '', text)      # remove hashtags
    new_text = re.sub(r'http\S+', '', new_text) # remove links
    new_text = re.sub(r'@\S+ ?', '', new_text)  # remove user names
    new_text = re.sub(r'\\x\w\w', '', new_text) # remove unicode chars
    new_text = re.sub(r'&amp', '', new_text)    # remove & code
    cleaned_data.append(new_text)




# write out cleaned up data to disk
with open('cleaned_tweets.txt', 'w') as myfile:
        for t in cleaned_data:
            myfile.write(str(t)+"\n")

