import ConfigParser, datetime, twitter, sys, os, glob, re
import pandas as pd
import numpy as np

#################################################
# Collect some tweeters and find their followers!
#################################################

# To Do:
# account for users following >5k.
# other independent variables?

def getMostFollowedAccounts():
    # top 1000 followed accounts. courtesy of http://top-twitter-users.silk.co/explore
    fil = '{0}/twitter_party/data/top_1000.csv'.format(os.path.expanduser("~"))
    df = pd.read_csv(fil,delimiter=',')
    
    # return as id:sn key pair
    id_sn = convertToIds(df.Twitter_Handle.tolist())
    return id_sn

def subsetDict(d):
    tooPolitical = ['realDonaldTrump','BarackObama','maddow','FoxNews','billmaher','HillaryClinton','StephenAtHome','MichelleObama','WhiteHouse','TheDailyShow','billclinton']
    d2 = dict(d)
    for key in d.keys():
        if d[key] in tooPolitical:
            del d2[key]
    return d2
    
def tweetLogIn():
    config = ConfigParser.ConfigParser()
    config.read('{}/.python_keys.conf'.format(os.path.expanduser("~")))
    t = twitter.Twitter(auth=twitter.OAuth(token=config.get('twitter','token'), token_secret=config.get('twitter','token_secret'), consumer_key=config.get('twitter','consumer_key'), consumer_secret=config.get('twitter','consumer_secret')),
        retry=True)
    return t

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
    return df
    
def storeSNs(df): 
    # Creates file stores SN, party (0,1,3), and term
    fil = '{0}/twitter_party/raw/sns.csv'.format(os.path.expanduser("~"))
    if os.path.isfile(fil):
        dfOld = pd.read_csv(fil,delimiter=',')
        df2 = df.append(dfOld, ignore_index=True)
        dfNew = df2.drop_duplicates('SN')
    else:
        dfNew = df
    dfNew.to_csv(fil, index=False)


def getFollowers(): 
    t = tweetLogIn()
    # use ID's since it's much faster apparently, and then just convert the relevant ones to screen names.
    # get's 5k id's at a time. Let's assume that's enough for v0.
    
    fil = '{0}/twitter_party/raw/sns.csv'.format(os.path.expanduser("~"))
    df = pd.read_csv(fil,delimiter=',')
    
    for index, row in df.iterrows():
        fil2 = '{0}/twitter_party/raw/{1}.csv'.format(os.path.expanduser("~"),row['SN'])
        if not os.path.isfile(fil2):
            try:
	    	tmp = t.friends.ids(screen_name=row['SN'])
            	dftemp = pd.DataFrame({'ids':tmp['ids']})
            	dftemp.to_csv(fil2, header=False,index=False)
	    except:
		print "Fail for: ", row


def convertToScreenNames(ids): 
    # take a list of ids and return a dict with screen names 
    # Note that id's may be banned/etc.
    t = tweetLogIn()
    
    SNs = {}
    limit = 100
    id_chunks = [ids[x:x+limit] for x in xrange(0, len(ids), limit)]
    for id_sub in id_chunks:
        tmp = t.users.lookup(user_id=','.join(str(i) for i in id_sub))
        for x in tmp:
            SNs[x['id']] = x['screen_name']
    return SNs


def convertToIds(sn_list): 
    # take a list of sns and return a dict with ids
    # Note that id's may be banned/etc.
    t = tweetLogIn()

    SNs = {}
    limit = 100
    sn_chunks = [sn_list[x:x+limit] for x in xrange(0, len(sn_list), limit)]
    for sn_sub in sn_chunks:
        tmp = t.users.lookup(screen_name=','.join(str(i) for i in sn_sub))
        for x in tmp:
            SNs[x['id']] = x['screen_name']
            #SNs.append(x['screen_name'])
    return SNs


def finalPolitical():
    #take political tweeters and only use ones where followers identified
    csvs = glob.glob("{}/twitter_party/raw/*.csv".format(os.path.expanduser("~")))
    regex = re.compile('raw/(.*)\.csv')
    sns = [m.group(1) for l in csvs for m in [regex.search(l)] if m]

    # match to party/test
    file = "{}/twitter_party/raw/sns.csv".format(os.path.expanduser("~"))
    df = pd.read_csv(file,delimiter=',',index_col=0,header=0)

    df = df.loc[df['Party'].isin([0,1])]

    # limit to accounts with followers procured
    df_2 = df[df.index.isin(sns)]
    
    # downsample to get even party count
    # df_2.Party.value_counts()
    if sum(df_2['Party']==1) > sum(df_2['Party']==0):
        less = 0
        more = 1
    else:
        less = 1
        more = 0        
    indices = np.where(df_2[['Party']] == more)[0]
    rng = np.random
    rng.shuffle(indices)
    n = sum(df_2['Party']==less)
    df_2 = df_2.drop(df_2.index[indices[n:]])
    # df_2.Party.value_counts()

    return df_2


def finalTest():
    # take non-political tweeters and only use ones where followers identified
    # could specify a term
    csvs = glob.glob("{}/twitter_party/raw/*.csv".format(os.path.expanduser("~")))
    regex = re.compile('raw/(.*)\.csv')
    sns = [m.group(1) for l in csvs for m in [regex.search(l)] if m]

    # match to party/test
    file = "{}/twitter_party/raw/sns.csv".format(os.path.expanduser("~"))
    df = pd.read_csv(file,delimiter=',',index_col=0,header=0)

    df = df.loc[df['Party'].isin([3])]

    # limit to accounts with followers procured
    df_2 = df[df.index.isin(sns)]

    return df_2

    

def mostFollowedAccounts(tweeters='political'):
    # For a set of tweeters (either political or test), find the most followed accounts
    assert tweeters in ['political','test'], "Undefined group {}".format(tweeters)


    if tweeters=='political':
        df = finalPolitical()
    elif tweeters=='test':
        df = finalTest()
    else:
        sys.exit("Undefined group {}".format(tweeters))
    
    filenames = [ "{}/twitter_party/raw/{}.csv".format(os.path.expanduser("~"),f) for f in df.index.tolist() ]
    
    # this ignores empty files
    ids_followed = None
    for f in filenames:
        A = np.genfromtxt(f,delimiter=',',dtype='|S32')
        if ids_followed is None:
            ids_followed = A
        else:
            try:ids_followed = np.concatenate([ids_followed,A])
            except:pass

    unique, counts = np.unique(ids_followed, return_counts=True)
    id_counts_followed = pd.DataFrame({'ids':unique,'counts':counts})
    
    limit = 1800
    id_sorted_followed = id_counts_followed.sort_values(by=['counts'],ascending=False)[0:limit]
    followed = convertToScreenNames(id_sorted_followed['ids'].tolist())
    return df, followed

def setupFollowerDataset():
    # running test and train separately gives different order to column names
    tweeters_political,followed_political = mostFollowedAccounts(tweeters='political')
    tweeters_test, followed_test = mostFollowedAccounts(tweeters='test')
    return tweeters_political,followed_political, tweeters_test, followed_test
    
    
def buildFollowerDataset(tweeters,followed):
    # To build dataset for training or prediction.
    # need tweeters (df index) and followers (columns)
    filenames = [ "{}/twitter_party/raw/{}.csv".format(os.path.expanduser("~"),f) for f in tweeters.index.tolist() ]
    
    # create input dataframe utweeters='Political'sing just these n sn's
    df = pd.DataFrame(columns=followed.values(), index=tweeters.index.tolist())
    for i, f in enumerate(filenames):
        temp = np.genfromtxt(f,delimiter=',',dtype='|S32')
        inds = np.where(np.in1d(np.array(followed.keys()),temp))[0]
        row = np.zeros(len(followed))
        row[inds] = 1
        df.iloc[i] = row

    assert df.isnull().values.ravel().sum()==0, "Processed dataframe contains Nulls"
    return df


def saveProcessedData(df,filename): 
    df.to_csv("{}/twitter_party/data/{}.csv".format(os.path.expanduser("~"),filename), index=True)


def identifyTestAccounts(Term,count=100):
    # Get list of test Accounts and store them in a file.
    testAccounts = getTweeters([Term],3,count)
    storeSNs(testAccounts)

def identifyPoliticalAccounts(count=100):
    # Get list of political Accounts and store them in a file.
	# What terms should I use?
    # this definition is very arbitrary! Something unsupervised might be more robust.
    # political/idealogical/issue/cultural?
    # 'liberal','democrat', 'republican','trump', 'gop'
    demFlags = ['resist','theresistance','singlepayer']
    repFlags = ['maga','tcot','prolife']
    
    demAccounts = getTweeters(demFlags,1,count)
    repAccounts = getTweeters(repFlags,0,count)
    
    storeSNs(demAccounts)
    storeSNs(repAccounts)


if __name__ == '__main__':
	# while testing:
	# t.application.rate_limit_status()
	
    # get some accounts and put them in a file
    #identifyPoliticalAccounts(count=100)
    
    # for each account, make a file with a list of their followers' ids
    #getFollowers()
    
    
    followed_all = getMostFollowedAccounts()
    followed_sub = subsetDict(followed_all)

    # now get most common followers from users and create a dataset for modeling.
    tweeters_political,followed_political, tweeters_test, followed_test = setupFollowerDataset()

    
    df = buildFollowerDataset(tweeters_political,followed_political)
    saveProcessedData(df,'dataframe_political_train')
    df_test = buildFollowerDataset(tweeters_test,followed_political)
    saveProcessedData(df_test,'dataframe_political_test')
    
    
    #df_test = buildFollowerDataset(tweeters_test,followed_test)
    #saveProcessedData(df_test,'dataframe_test_train')
    #df = buildFollowerDataset(tweeters_political,followed_test)
    #saveProcessedData(df,'dataframe_test_test')
    
    df = buildFollowerDataset(tweeters_political,followed_sub)
    saveProcessedData(df,'dataframe_sub_train')
    df_test = buildFollowerDataset(tweeters_test,followed_sub)
    saveProcessedData(df_test,'dataframe_sub_test')
    
    df = buildFollowerDataset(tweeters_political,followed_all)
    saveProcessedData(df,'dataframe_top_train')
    df_test = buildFollowerDataset(tweeters_test,followed_all)
    saveProcessedData(df_test,'dataframe_top_test')

