import ConfigParser, datetime, twitter, sys, os
import pandas as pd

#############
# Collect some tweets!
#######

# To Do:
# account for users following >5k. 
# parse >100 id - sn requests

def getTweeters(terms):
    # Return screen names from users tweeting on these terms!
    # I want tweets with identifying twitter feeds
    # in english and in US.

    t = tweetLogIn()
    
    # each tweet has 
    # 'text' - obvious value
    # 'source' - which could be iphone
    # 'time_zone' which could identify US - #bigTweetList['statuses'][k]['user']['time_zone']
    # 'location' - which is ambiguous
    # 'lang' - 'en'
    # 'description' - profile
    # screen_name
    
    SNs = []
    for d in terms:
        bigTweetList = t.search.tweets(q="#"+d,count=100)
        for k in xrange(len(bigTweetList['statuses'])):
            if bigTweetList['statuses'][k]['lang']=='en':
                SNs.append(bigTweetList['statuses'][k]['user']['screen_name'])
                #print bigTweetList['statuses'][k]['text'] 

    # make lists unique.
    # use a small list to identify key variables - followers. other things?
    
    SNs = list(set(SNs))
    return SNs
    

def tweetLogIn():
    config = ConfigParser.ConfigParser()
    config.read('{}/.python_keys.conf'.format(os.path.expanduser("~")))
    t = twitter.Twitter(auth=twitter.OAuth(token=config.get('twitter','token'), token_secret=config.get('twitter','token_secret'), consumer_key=config.get('twitter','consumer_key'), consumer_secret=config.get('twitter','consumer_secret')),
        retry=True)
    return t

def tweetCollect():

    t = tweetLogIn()

    #t.statuses.update(status='a tweet')

def tweetUserCollect(user='Hasty_Data'):
    # go get some tweets from a specific user
    t = tweetLogIn()
    t.statuses.user_timeline(screen_name=user)

def tweetMyTimeline():
    t = tweetLogIn()
    temp = t.statuses.home_timeline()

def getFollowers(screennames): 
    t = tweetLogIn()
    # use ID's since it's much faster apparently, and then just convert the relevant ones to screen names.
    # get's 5k id's at a time. Let's assume that's enough for v0.
    friends={}
    for sn in screennames:
        tmp = t.friends.ids(screen_name=sn)
        friends[sn] = tmp['ids']

    return friends

def convertToScreenNames(ids): 
    t = tweetLogIn()

    # right now, take a list and return a list.
    # If List>100 need to parse out I think
    SNs = []
    tmp = t.users.lookup(user_id=','.join(str(i) for i in ids))
    for x in tmp:
        SNs.append(x['screen_name'])

    return SNs



if __name__ == '__main__':
	# while testing:
	t.application.rate_limit_status()
	
	# Find terms that identify individuals
    # this definition is very arbitrary! Something unsupervised might be more robust.
    # political/idealogical/issue/cultural?
    
    # 'liberal','democrat', 'republican','trump', 'gop'
    demFlags = ['resist','theresistance','hrc','singlepayer']
    repFlags = ['maga','tcot','prolife']
    
    demAccounts = getTweeters(demFlags)
    repAccounts = getTweeters(repFlags)
    
    # get who these users follow
    # I get ~1 request per minute? That seems pretty restrictive?
    #test = ['Hasty_Data']
    friendsD = getFollowers(demAccounts)
    friendsR = getFollowers(repAccounts)
    
    # trim this list of ids 
    
    ids = friends['Hasty_Data']

    # then convert the id's to sn's
    convertToScreenNames(something)
    
    # then convert this to a dataframe as observation and features.
    
    