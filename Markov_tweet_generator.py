

# http://github.com/timestocome


# import other libs 
# https://github.com/jsvine/markovify
import markovify                    

#######################################################################
# read in text and generate text
#######################################################################

filename = 'RomanceScamEmails.txt'

def Markov_tweet_generator():

    file = open('RomanceScamEmails.txt', encoding='utf-8')
    data = file.read()
    file.close()

    # create markov model
    model_3 = markovify.Text(data, state_size=3)

    # generate text from model
    sentences = []
    print("*******************************")
    for i in range(200):
        s = model_3.make_sentence()
        if s != None:               # dud
            if len(s) <=140:        # too long for twitter
                sentences.append(s)


    # remove duplicates
    sentences = list(set(sentences))

    # test to see if it's working
    #print("Good tweets", len(sentences))
    #for i in sentences:
    #    print(i)

    return sentences

##########################################
#Markov_tweet_generator()