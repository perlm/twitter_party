import ConfigParser, datetime, twitter, sys, os, glob, re
import pandas as pd
import numpy as np


#############
# Collect some tweets!
#######

# To Do:
# account for users following >5k and 0.
# parse >100 id - sn requests (if necessary)
# pep8?


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

def storeSNs(df): 
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
    
    storeSNs(demAccounts)
    storeSNs(repAccounts)


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
    # take a list of ids and return a dict with screen names
    # One of the top 500 followed accounts actuallly no longer exists
    # need to be able to account for this.
    t = tweetLogIn()
    
    # If >100 need to split up
    SNs = {}
    limit = 100
    id_chunks = [ids[x:x+limit] for x in xrange(0, len(ids), limit)]

    for id_sub in id_chunks:
        tmp = t.users.lookup(user_id=','.join(str(i) for i in id_sub))
        for x in tmp:
            SNs[x['id']] = x['screen_name']
            #SNs.append(x['screen_name'])

    return SNs


def buildFollowerDataset(): 
    filenames = glob.glob("{}/twitter_party/raw/*.csv".format(os.path.expanduser("~")))
    filenames.remove("{}/twitter_party/raw/sns.csv".format(os.path.expanduser("~")))
    
    # get user screen name from filename
    regex = re.compile('raw/(.*)\.csv')
    sns = [m.group(1) for l in filenames for m in [regex.search(l)] if m]

    # get most commonly followed id's
    ids_followed = np.concatenate([np.genfromtxt(f,delimiter=',',dtype='|S32') for f in filenames])
    unique, counts = np.unique(ids_followed, return_counts=True)
    id_counts_followed = pd.DataFrame({'ids':unique,'counts':counts})
    
    # users.lookup can do 900 before limit.
    limit = 900
    id_sorted_followed = id_counts_followed.sort_values(by=['counts'],ascending=False)[0:limit]
    #id_np_followed = np.array(id_sorted_followed['ids']) #I want to limit it to those with valid sn
    sn_followed = convertToScreenNames(id_sorted_followed['ids'].tolist())
    id_np_followed = np.array(sn_followed.keys())
    
    # create input dataframe
    df = pd.DataFrame(columns=sn_followed.values(), index=sns)
    for i, f in enumerate(filenames):
        temp = np.genfromtxt(f,delimiter=',',dtype='|S32')
        inds = np.where(np.in1d(id_np_followed,temp))[0]
        row = np.zeros(len(sn_followed))
        row[inds] = 1
        df.iloc[i] = row

    return df


if __name__ == '__main__':
	# while testing:
	# t.application.rate_limit_status()
	
    # get some accounts and put them in a file
    identifyPoliticalAccounts(count=100)
    
    # for each account, make a file with a list of their followers' ids
    getFollowers()
    
    # now get most common followers from users and create a dataset for modeling.
    df = buildFollowerDataset()
    

