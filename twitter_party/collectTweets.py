import ConfigParser, datetime, twitter, sys, os
import pandas as pd

#############
# Collect some tweets!
#######

def getScreenNames():
    # go get some tweets!
    # I want tweets with identifying twitter feeds
    # in english and in US.

    t = tweetLogIn()
    
    # this definition is very arbitrary! Something unsupervised might be more robust.
    # political/idealogical/issue/cultural?
    
    # 'liberal','democrat', 'republican','trump', 'gop'
    demFlags = ['resist','theresistance','hrc','singlepayer']
    repFlags = ['maga','tcot','prolife']

    demAccounts = []
    repAccounts = []
    # each tweet has 
    # 'text' - obvious value
    # 'source' - which could be iphone
    # 'time_zone' which could identify US - #bigTweetList['statuses'][k]['user']['time_zone']
    # 'location' - which is ambiguous
    # 'lang' - 'en'
    # 'description' - profile
    # screen_name
    # 
    
    for d in demFlags:
        bigTweetList = t.search.tweets(q="#"+d,count=100)
        for k in xrange(len(bigTweetList['statuses'])):
            if bigTweetList['statuses'][k]['lang']=='en':
                demAccounts.append(bigTweetList['statuses'][k]['user']['screen_name'])
                #print bigTweetList['statuses'][k]['text'] 

    for d in repFlags:
        bigTweetList = t.search.tweets(q="#"+d,count=100)
        for k in xrange(len(bigTweetList['statuses'])):
            if bigTweetList['statuses'][k]['lang']=='en':
                repAccounts.append(bigTweetList['statuses'][k]['user']['screen_name'])
                #print bigTweetList['statuses'][k]['text'] 

    # make lists unique.
    # use a small list to identify key variables - followers. other things?
    
    demAccounts = list(set(demAccounts))
    repAccounts = list(set(repAccounts))
    
    return demAccounts, repAccounts
    

def tweetLogIn():
    config = ConfigParser.ConfigParser()
    config.read('{}/.python_keys.conf'.format(os.path.expanduser("~")))
    t = twitter.Twitter(auth=twitter.OAuth(token=config.get('twitter','token'), token_secret=config.get('twitter','token_secret'), consumer_key=config.get('twitter','consumer_key'), consumer_secret=config.get('twitter','consumer_secret')))
    return t

def tweetCollect():

    t = tweetLogIn()

    #t.statuses.update(status='a tweet')

def tweetUserCollect(user='Hasty_Data'):
    # go get some tweets from a specific user
    t = tweetLogIn()
    t.statuses.user_timeline(screen_name=user)

def tweetMyFeed():
    # go get some tweets from a specific user
    t = tweetLogIn()
    temp = t.statuses.home_timeline()


if __name__ == '__main__':
	#print "Hey now!"

    demAccounts, repAccounts = getScreenNames()

