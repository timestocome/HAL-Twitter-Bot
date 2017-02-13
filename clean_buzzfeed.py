# http://github.com/timestocome

# Buzzfeed data sets are located at 
# https://github.com/BuzzFeedNews


############################################################################
# setup
#############################################################################
# import standard libs
import numpy as np
import string as st 


filename = 'buzzfeed.txt'

#######################################################################
# read in csv data, titles only, and clean them up
#######################################################################
def Cleanup_titles():
    data = []
    with open(filename) as f:
        for line in f:
            data.append(line)
    

    
    from itertools import chain
    import re

    cleaned_data = []
    for text in data:
        new_text = re.sub(r"(\w)([A-Z])", r"\1 \2", text)
        new_text = re.sub(r"(\"\")", " ", new_text)

        print(new_text)
        cleaned_data.append(new_text)


    # write out cleaned up data to disk
    with open('cleaned_buzzfeed.txt', 'w') as myfile:
        for t in cleaned_data:
            myfile.write(str(t)+"\n")


###########################################################################
#
#########################################################################            


########################################################################
Cleanup_titles()