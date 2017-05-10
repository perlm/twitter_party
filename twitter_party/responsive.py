import ConfigParser, twitter
import numpy as np
import pandas as pd
from sklearn import preprocessing, feature_extraction, linear_model, metrics, model_selection, ensemble
import math, glob, datetime, os, pickle

from collectTweets import *
from buildModel import *


#################################################
# Respond to tweets with models!
#################################################

def getAts():
    t = tweetLogIn()
    ats=[]
    out = t.statuses.mentions_timeline()
    for k in xrange(len(out)):
        #print out[k]['user']['screen_name']
        if any(substring in out[k]['text'].lower() for substring in ['party','partisanship','political']):
            ats.append(out[k]['user']['screen_name'])

    ats = list(set(ats))
    df_ats = pd.DataFrame({'users':ats,'score':0})
    fil = '{}/twitter_party/log/responses.log'.format(os.path.expanduser("~"))
    #df_ats.to_csv(fil, header=True,index=False) # to initialize

    old = pd.read_csv(fil,delimiter=',',header=0)
    new = df_ats[~df_ats['users'].isin(old['users'])].dropna()
    
    return new

def getPredictFollowed(sn):
    t = tweetLogIn()
    tmp = t.friends.ids(screen_name=sn)
    return pd.DataFrame({'ids':tmp['ids']})

def sendTweet(p,name):
    t = tweetLogIn()
    
    if 0<=p<0.3:q = "The model is pretty sure you're conservative"
    elif 0.3<=p<0.7:q = "The model gives you even odds for conservative/liberal"
    elif 0.7<=p<=1.0:q = "The model is pretty sure you're liberal"
    
    text = "@" + name + ". " + q + " ("+ str(int(p*100)) + "% prob Liberal) https://hastydata.wordpress.com/2017/05/07/twitter-party/" 
    print text
    t.statuses.update(status=text)

if __name__ == '__main__':
    model_1 = pickle.load(open("{}/twitter_party/model_pickles/political.model".format(os.path.expanduser("~")),"r"))    
    model_2 = pickle.load(open("{}/twitter_party/model_pickles/sub1000.model".format(os.path.expanduser("~")),"r"))
    scaler_1 = pickle.load(open("{}/twitter_party/model_pickles/political.scaler".format(os.path.expanduser("~")),"r"))    
    scaler_2 = pickle.load(open("{}/twitter_party/model_pickles/sub1000.scaler".format(os.path.expanduser("~")),"r"))

    df_followers1 = readFollowerData('dataframe_political_train')
    df_followers2 = readFollowerData('dataframe_sub_train')
    
    followers1 = df_followers1.columns.values
    followers2 = df_followers2.columns.values
    followers1 = convertToIds(followers1.tolist())
    followers2 = convertToIds(followers2.tolist())
    
    fil = '{}/twitter_party/log/responses.log'.format(os.path.expanduser("~"))
    
    # get new 
    ats = getAts()

    prob=None
    for iii, r in ats.iterrows():
        # get followers
        f = getPredictFollowed(r['users'])
        
        # transform into model input and predict!
        df = pd.DataFrame(columns=followers1.values(), index=[r['users']])
        inds = np.where(np.in1d(np.array(followers1.keys()),np.array(f['ids'])))[0]
        row = np.zeros(len(followers1))
        row[inds] = 1
        df.iloc[0] = row
        X, X_scaled, Y, scaler,X_fix = processData(df,scaler_1)
        prob1 = predict(X_scaled,model_1)
        
        # use prob1 if very large/small
        if prob1>0.8 or prob1<0.2:
            prob=prob1
        else:
            df = pd.DataFrame(columns=followers2.values(), index=[r['users']])
            inds = np.where(np.in1d(np.array(followers2.keys()),np.array(f['ids'])))[0]
            row = np.zeros(len(followers2))
            row[inds] = 1
            df.iloc[0] = row
            X, X_scaled, Y, scaler,X_fix = processData(df,scaler_2)
            prob = predict(X_scaled,model_2)
    
        # tweet
        sendTweet(prob[0],r['users'])
        
        # log
        app = pd.DataFrame({'score':[prob[0]],'users':[r['users']]})
        app.to_csv(fil, header=False,index=False,mode='a')
    
