import numpy as np
import pandas as pd
from sklearn import preprocessing, feature_extraction, linear_model, metrics, model_selection, ensemble
import math, glob, datetime, os, pickle
import matplotlib.pyplot as plt
plt.style.use('ggplot')


####
# This file contains functions for building the classification model.
###

def optimizeLambdaLogistic(X_train, X_test, y_train, y_test,L='l1'):
	# use CV to optimize regularization hyperparameter! (using either L1 or L2) (lambda is inverse C here)

	if L=='l1':
		tuned_parameters = [ {'C':[1e-5,1e-3,5e-3,7.5e-3,1e-2,2.5e-2,5e-2,1e-1,5e-1,1e0,1e8]}]
	else:
		tuned_parameters = [ {'C':[1e-8,1e-6,1e-4,1e-2, 1e0,1e2,1e4,1e6,1e8]}]

	clf = model_selection.GridSearchCV(linear_model.LogisticRegression(penalty=L), tuned_parameters, cv=50,scoring='roc_auc')
	clf.fit(X_train, y_train)

	print "Hyperparameter Optimization, penalty=", L
	print(clf.best_params_)

	means = clf.cv_results_['mean_test_score']
	stds = clf.cv_results_['std_test_score']
	#for mean, std, params in zip(means, stds, clf.cv_results_['params']):print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

	y_prob = clf.predict_proba(X_test)[:,1]
	y_class = clf.predict(X_test)
	#print y_test, y_prob, y_class
	#print "Hyperparameter Optimization"
	#print(metrics.classification_report(y_test, y_class))

	return clf.best_params_


def buildLogisticModel(X_scaled,Y,X_fix,optimize=True):
	# build a model! l1 for lasso, l2 for ridge
	# use CV and holdout.
	X_train, X_test, y_train, y_test = model_selection.train_test_split(X_scaled, Y, test_size=0.3, random_state=0)

	# need to reshape for some reason...
	Y = Y.as_matrix()
	c, r = Y.shape
	Y = Y.reshape(c,)

	y_train = y_train.as_matrix()
	c, r = y_train.shape
	y_train = y_train.reshape(c,)

	y_test = y_test.as_matrix()
	c, r = y_test.shape
	y_test = y_test.reshape(c,)

	if optimize:
	    # optimize hyperparameter	
	    la = optimizeLambdaLogistic(X_train, X_test, y_train, y_test,'l1')
	    #lb = optimizeLambdaLogistic(X_train, X_test, y_train, y_test,'l2')	# untested

	    # train model using hyperparameter
	    model = linear_model.LogisticRegression(C=la['C'], penalty='l1')
	else:
	    model = linear_model.LogisticRegression(C=0.05, penalty='l1')
	model.fit(X_train,y_train)

	y_prob = model.predict_proba(X_test)[:,1]
	y_class = model.predict(X_test)
	print "Final Model: Out of Sample Performance"
	print(metrics.classification_report(y_test, y_class))

	print "AUC:", metrics.roc_auc_score(y_test, y_prob)

	# retrain on whole data set.
	if optimize:
		model = linear_model.LogisticRegression(C=la['C'], penalty='l1')
	else:
		model = linear_model.LogisticRegression(C=0.05, penalty='l1')
	model.fit(X_scaled,Y)

	print model.intercept_
	factors = list(X_fix.columns.values)
	coefs 	= list(model.coef_.ravel())
	for i,f in enumerate(factors):
	    if coefs[i]!=0:
		    print f,"\t", coefs[i]

	return model


def buildRandomForest(X_scaled,Y,X_fix,W=None):
	########################
	# try a random forest model!
	##########################3
    
    if W is None:
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X_scaled, Y, test_size=0.3, random_state=0)
    else:
        X_train, X_test, y_train, y_test,w_train,w_test = model_selection.train_test_split(X_scaled, Y, W,test_size=0.3, random_state=0)
    
    # need to reshape for some reason...
    Y = Y.as_matrix()
    c, r = Y.shape
    Y = Y.reshape(c,)
    
    y_train = y_train.as_matrix()
    c, r = y_train.shape
    y_train = y_train.reshape(c,)
    
    y_test = y_test.as_matrix()
    c, r = y_test.shape
    y_test = y_test.reshape(c,)

    if W is not None:
        w_train = w_train.as_matrix()
        w_test = w_test.as_matrix()
    
    rf = ensemble.RandomForestClassifier(n_estimators=1000)

    if W is None:
        rf.fit(X_train, y_train)
    else:
        rf.fit(X_train, y_train,sample_weight=w_train)
    
    y_prob = rf.predict_proba(X_test)[:,1]
    y_class = rf.predict(X_test)
    print "Final RF Model: Out of Sample Performance"
    print(metrics.classification_report(y_test, y_class))
    print "AUC:", metrics.roc_auc_score(y_test, y_prob)
    
    # retrain on whole data set.
    if W is None:
        rf.fit(X_scaled, Y)
    else:
        rf.fit(X_scaled, Y,sample_weight=W.as_matrix())
    
    importances = list(rf.feature_importances_)
    factors = list(X_fix.columns.values)
    deets = pd.DataFrame({'importances':importances,'factors':factors})
    print deets.sort_values(by=['importances'],ascending=[False]).head(10)
    #for i,f in enumerate(factors):
    #    print f,"\t", importances[i]
    return rf


def predict(X_scaled,model):
	# for a given model and independent data, generate predictions
	y_prob = model.predict_proba(X_scaled)[:,1]
	print "Avg of predictions= ", np.mean(y_prob)
	return y_prob

	
def processData(df,scaler=None):
    #################3
	# take dataframe and reformat for sci-kit learn
    ################

    # Assume that "population" is 50-50.   
    # df.groupby('Party').size() 
    # df.Party.value_counts()
    # shows that the sample is not, so downsample to get to 50/50.
    # this assumes 0>1 in sample. could make more robust? or just add if statement.

    # downsample
    #if sum(df['Party']==1) > sum(df['Party']==0):
    #    less = 0
    #    more = 1
    #else:
    #    less = 1
    #    more = 0        
    #indices = np.where(df[['Party']] == more)[0]
    #rng = np.random
    #rng.shuffle(indices)
    #n = sum(df['Party']==less)
    #df_downsample = df.drop(df.index[indices[n:]])
    df_downsample = df
    # df_downsample.Party.value_counts()

    # for features, just use whether or not they follow the top n accounts
    cols = [col for col in df_downsample.columns if col not in ['Term', 'Party','followed','weight']]
    X = df_downsample[cols]
    
    # if using multi-level factors - (just dummies for now)
    # X_fix = pd.get_dummies(X)
    X_fix = X
    
    Y = df_downsample[['Party']]

    # To skip scaling
    # X_scaled = X_fix
    
    # scale features
    if scaler is None:
        scaler = preprocessing.StandardScaler().fit(X_fix)	#this allows me to re-use the scaler.
        X_scaled = scaler.transform(X_fix) 
    else:
        X_scaled = scaler.transform(X_fix) 
    
    return X,X_scaled, Y, scaler, X_fix

def readDependentData():
    file = "{}/twitter_party/raw/sns.csv".format(os.path.expanduser("~"))
    df = pd.read_csv(file,delimiter=',',index_col=0,header=0)
    return df
	
def readFollowerData(filename):
    file = "{}/twitter_party/data/{}.csv".format(os.path.expanduser("~"),filename)
    df = pd.read_csv(file,delimiter=',',index_col=0,header=0)
    return df

def addrows(df,df_predict):
    return pd.concat([df,df_predict])


def model_covariate_shift():
    df_followers = readFollowerData('dataframe_political_train')
    df_test_followers = readFollowerData('dataframe_political_test')
    df_dependent = readDependentData()
    df = df_dependent.join(df_followers,how='inner')
    df_test = df_dependent.join(df_test_followers,how='inner')
    df_all = addrows(df,df_test)
    df_all['followed'] = df_all.sum(axis=1) - df_all['Party']
    
    forPlot = df_all.pivot(columns='Party',values='followed')
    ax = forPlot.plot.hist(bins=25,stacked=True,normed=1,alpha=0.75,title='Twitter Party Model - Covariate Shift')
    ax.set_xlabel("Number of Input Accounts Followed")

    df_all = df_all.drop('followed', 1)

    df['followed'] = df.sum(axis=1) - df['Party']
    df_test['followed'] = df_test.sum(axis=1) - df_test['Party']

    # calc weighting using 1d # of accounts followed.
    t1 = df.groupby(['followed']).size()/len(df)
    t2 = df_test.groupby(['followed']).size()/len(df_test)
    t3 = t2/t1
    t3 = t3.fillna(0)
    
    df_weighted = df.merge(t3.to_frame(),left_on='followed',right_index=True)
    df_weighted.columns.values[-1] = 'weight'
    df_weighted.rename(columns={0:'weight'}, inplace=True)

    X, X_scaled, Y, scaler,X_fix = processData(df_weighted)
    model_rf = buildRandomForest(X_scaled,Y,X_fix,df_weighted['weight'])
    
    df_weighted['y_probs_rf'] = predict(X_scaled,model_rf)
    
    # distribution by term
    forPlot = df_weighted.pivot(columns='Term',values='y_probs_rf')
    ax = forPlot.plot.hist(bins=25,stacked=True,normed=1,alpha=0.75,title='Twitter Party Model - Covariate Shift Adaptation')
    ax.set_xlabel("Predicted Probability Democrat")

    X, X_scaled, Y, scaler,X_fix = processData(df_all,scaler)
    df_all['y_probs_rf'] = predict(X_scaled,model_rf)
    df_new = df_all.loc[df_all['Party']==3]

    df_new['followed'] = df_new.sum(axis=1) - df_new['y_probs_rf'] - df_new['Party']
    df_subset = df_new.loc[df_new['followed']>0]

    # distribution by term
    forPlot = df_subset.pivot(columns='Term',values='y_probs_rf')
    ax = forPlot.plot.hist(bins=25,stacked=True,normed=1,alpha=0.75,title='Twitter Party Model - Test Tweeters')
    ax.set_xlabel("Predicted Probability Democrat")
        
        # boxplot
        ax = df_subset.boxplot(column='y_probs_rf',by='Term')
        ax.set_xlabel("Tweeter keyword")
        ax.set_ylabel("Predicted Probability Democrat")
        ax.set_title("")


def original_model_workflow():    
    # load in summary file with screen names
    # and merge it to file with processed follower data

    df_followers = readFollowerData('dataframe_political_train')
    df_test_followers = readFollowerData('dataframe_political_test')
    
    #df_followers = readFollowerData('dataframe_sub_train')
    #df_test_followers = readFollowerData('dataframe_sub_test')
    
    #df_followers = readFollowerData('dataframe_top_train')
    #df_test_followers = readFollowerData('dataframe_top_test')

    df_dependent = readDependentData()
    df = df_dependent.join(df_followers,how='inner')
    X, X_scaled, Y, scaler,X_fix = processData(df)
    
    #model_lr = buildLogisticModel(X_scaled,Y,X_fix,optimize=True)
    #y_probs_lr = predict(X_scaled,model_lr)
    
    # df['followed'] = df.sum(axis=1) - df['Party']
    #1.0*len(df[(df['followed']>0)])/len(df)
    
    # rf is slightly more accurate, but slightly lower AUC, well not really.
    model_rf = buildRandomForest(X_scaled,Y,X_fix)
    df['y_probs_rf'] = predict(X_scaled,model_rf)

    # save model to file.
    #pickle.dump(model_rf,open("{}/twitter_party/data/subset1000.model".format(os.path.expanduser("~")),"w"))

    # histogram for funsies.
    while False:
        plt.hist(df['y_probs_rf'], 25, normed=1, facecolor='blue', alpha=0.75)
        ax = df['y_probs_rf'].plot(kind='hist',bins=50,normed=1,facecolor='blue',alpha=0.75,title='Twitter Party Model')
        ax.set_xlabel("Predicted Probability Democrat")
        df['y_probs_rf'].hist(by=df['Term'],sharex=True,bins=50,normed=1,facecolor='blue',alpha=0.75)
    
        forPlot = df.pivot(columns='Term',values='y_probs_rf')
        ax = forPlot.plot.hist(bins=25,stacked=True,normed=1,alpha=0.75,title='Twitter Party Model')
        ax.set_xlabel("Predicted Probability Democrat")
    
    # load in model.
    #model_rf = pickle.load(open("{}/twitter_party/data/model.model".format(os.path.expanduser("~")),"r"))
    
    df = df.drop('y_probs_rf', 1)
    
    df_test = df_dependent.join(df_test_followers,how='inner')
    df_all = addrows(df,df_test)
    X, X_scaled, Y, scaler,X_fix = processData(df_all,scaler)
    df_all['y_probs_rf'] = predict(X_scaled,model_rf)
    df_new = df_all.loc[df_all['Party']==3]
    
    # histogram for funsies.
    while False:
        df_new['followed'] = df_new.sum(axis=1) - df_new['y_probs_rf'] - 3.0
        1.0*len(df_new[(df_new['followed']>0)])/len(df_new)
        # 65% for political. 77% for sub
        
        df_subset = df_new.loc[df_new['followed']>0]
        #ax = df_subset['y_probs_rf'].plot(kind='hist',bins=25,xlim=[0,1],normed=1,facecolor='blue',alpha=0.75,title='Twitter Party Model - NBA Tweeters')
        #ax.set_xlabel("Predicted Probability Democrat")

        # distribution by term
        forPlot = df_subset.pivot(columns='Term',values='y_probs_rf')
        ax = forPlot.plot.hist(bins=25,stacked=True,normed=1,alpha=0.75,title='Twitter Party Model - Test Tweeters')
        ax.set_xlabel("Predicted Probability Democrat")
        
        # boxplot
        ax = df_subset.boxplot(column='y_probs_rf',by='Term')
        ax.set_xlabel("Tweeter keyword")
        ax.set_ylabel("Predicted Probability Democrat")
        ax.set_title("")
        
        
        fig=plt.figure()
        ax1=fig.add_subplot(111)
        ax1.scatter(df_new['followed'],df_new['y_probs_rf'],alpha=0.5,facecolor='blue')
        ax1.set_ylim([0,1])
        ax1.set_xlim([0,50])
        ax1.set_xlabel('Number of Followed in set')
        ax1.set_xlabel('Predicted Probability')


    df_test = df_dependent.join(df_test_followers,how='inner')
    df_all = addrows(df,df_test)
    X, X_scaled, Y, scaler,X_fix = processData(df_all,scaler)
    df_all['y_probs_rf_2'] = predict(X_scaled,model_rf)
    df_new_2 = df_all.loc[df_all['Party']==3]
    
    df_new['y_probs_rf_2'] = df_new_2['y_probs_rf_2']
    df_new['followed'] = df_new.sum(axis=1) - df_new_3['y_probs_rf'] - df_new_3['y_probs_rf_2']
        1.0*len(df_new[(df_new['followed']>0)])/len(df_new)
        # 65% for political. 77% for sub
        
        df_subset = df_new.loc[df_new['followed']>0]
    
    
    fig=plt.figure()
    ax1=fig.add_subplot(111)
    ax1.hexbin(df_new_3['y_probs_rf'],df_new_3['y_probs_rf_2'],gridsize=20)
    ax1.set_ylim([0,1])
    ax1.set_xlim([0,1])
    ax1.set_xlabel('Predicted Probability 1')
    ax1.set_ylabel('Predicted Probability 2')


    
if __name__ == '__main__':
    print "not set up"
