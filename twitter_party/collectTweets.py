import ConfigParser, datetime, twitter, sys, os
import pandas as pd

#############
# Collect some tweets!
#######

# To Do:
# account for users following >5k.
# write code to trim id list
# parse >100 id - sn requests (if necessary)
# clean up functions and naming and formatting.

def getTweeters(terms,party,count=100):
    # Given a list of terms
    # Return dataframe of screen names from users tweeting those terms
    # currently specifying English language but not US location

    t = tweetLogIn()
    
    # each tweet has 
    # 'text' - obvious value
    # 'source' - which could be iphone
    # 'time_zone' which could identify US - #bigTweetList['statuses'][k]['user']['time_zone']
    # 'location' - which is ambiguous
    # 'lang' - 'en'
    # 'description' - profile
    # screen_name
    
    df = pd.DataFrame(columns=['SN','Term','Party'])
    for d in terms:
        bigTweetList = t.search.tweets(q="#"+d,count=count)
        for k in xrange(len(bigTweetList['statuses'])):
            if bigTweetList['statuses'][k]['lang']=='en':
                
                if bigTweetList['statuses'][k]['user']['screen_name'] not in df.SN.values:
                    df = df.append({'SN':bigTweetList['statuses'][k]['user']['screen_name'],'Term':d,'Party':party},ignore_index=True)
                #SNs.append(bigTweetList['statuses'][k]['user']['screen_name'])
                #print bigTweetList['statuses'][k]['text'] 

    # make lists unique.
    # use a small list to identify key variables - followers. other things?
    
    #SNs = list(set(SNs))
    return df
    

def tweetLogIn():
    config = ConfigParser.ConfigParser()
    config.read('{}/.python_keys.conf'.format(os.path.expanduser("~")))
    t = twitter.Twitter(auth=twitter.OAuth(token=config.get('twitter','token'), token_secret=config.get('twitter','token_secret'), consumer_key=config.get('twitter','consumer_key'), consumer_secret=config.get('twitter','consumer_secret')),
        retry=True)
    return t

def store_SNs(df): 
    # since I'll get locked out a bunch, I might as well write down SNs
    # find last entry in data file
    
    fil = '{0}/twitter_party/raw/sns.csv'.format(os.path.expanduser("~"))
    if os.path.isfile(fil):
        dfOld = pd.read_csv(fil,delimiter=',')
        df2 = df.append(dfOld, ignore_index=True)
        dfNew = df2.drop_duplicates('SN')
    else:
        dfNew = df
    
    dfNew.to_csv(fil, index=False)

def identifyPoliticalAccounts(count=100):
    # Get list of political Accounts and store them in a file.
	
	# What terms should I use?
    # this definition is very arbitrary! Something unsupervised might be more robust.
    # political/idealogical/issue/cultural?
    # 'liberal','democrat', 'republican','trump', 'gop'
    demFlags = ['resist','theresistance','singlepayer']
    repFlags = ['maga','tcot','prolife']
    
    demAccounts = getTweeters(demFlags,1,count)
    repAccounts = getTweeters(repFlags,2,count)
    
    store_SNs(demAccounts)
    store_SNs(repAccounts)


def getFollowers(): 
    t = tweetLogIn()
    # use ID's since it's much faster apparently, and then just convert the relevant ones to screen names.
    # get's 5k id's at a time. Let's assume that's enough for v0.
    
    fil = '{0}/twitter_party/raw/sns.csv'.format(os.path.expanduser("~"))
    df = pd.read_csv(fil,delimiter=',')
    
    for index, row in df.iterrows():
        fil2 = '{0}/twitter_party/raw/{1}.csv'.format(os.path.expanduser("~"),row['SN'])
        if not os.path.isfile(fil2):
            tmp = t.friends.ids(screen_name=row['SN'])
            dftemp = pd.DataFrame({'ids':tmp['ids']})
            dftemp.to_csv(fil2, header=False,index=False)


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
	# t.application.rate_limit_status()
	
    # get some accounts and put them in a file
    identifyPoliticalAccounts(count=100)
    
    # for each account, make a file with a list of their followers' ids
    getFollowers()
    
    # get who these users follow
    # I get ~1 request per minute? That seems pretty restrictive?
    
    # trim this list of ids to one's that occur most frequently
    # trim_in_some_way()
    # then convert the id's to sn's
    #convertToScreenNames(something)
    
    # then convert this to a dataframe as observation and features.
    
    