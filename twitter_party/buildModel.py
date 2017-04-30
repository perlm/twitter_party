import numpy as np
import pandas as pd
from sklearn import preprocessing, feature_extraction, linear_model, metrics, model_selection, ensemble
import math, glob, datetime, os, pickle
import matplotlib.pyplot as plt


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


def buildRandomForest(X_scaled,Y,X_fix):
	########################
	# try a random forest model!
	##########################3
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
    
    rf = ensemble.RandomForestClassifier(n_estimators=1000)
    rf.fit(X_train, y_train)
    
    y_prob = rf.predict_proba(X_test)[:,1]
    y_class = rf.predict(X_test)
    print "Final RF Model: Out of Sample Performance"
    print(metrics.classification_report(y_test, y_class))
    print "AUC:", metrics.roc_auc_score(y_test, y_prob)
    
    # retrain on whole data set.
    rf.fit(X_scaled, Y)
    
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
    if sum(df['Party']==1) > sum(df['Party']==0):
        less = 0
        more = 1
    else:
        less = 1
        more = 0        
    indices = np.where(df[['Party']] == more)[0]
    rng = np.random
    rng.shuffle(indices)
    n = sum(df['Party']==less)
    df_downsample = df.drop(df.index[indices[n:]])

    # df_downsample.Party.value_counts()

    # for features, just use whether or not they follow the top n accounts
    cols = [col for col in df_downsample.columns if col not in ['Term', 'Party']]
    X = df_downsample[cols]
    
    # if using multi-level factors - (just dummies for now)
    # X_fix = pd.get_dummies(X)
    X_fix = X
    
    Y = df_downsample[['Party']]

    # scale features
    #if scaler is None:
    #    scaler = preprocessing.StandardScaler().fit(X_fix)	#this allows me to re-use the scaler.
    #    X_scaled = scaler.transform(X_fix) 
    #else:
	#    X_scaled = scaler.transform(X_fix) 
    
    # since I'm only using dummies, I'm going to leave them unscaled.
    X_scaled = X_fix
    return X,X_scaled, Y, scaler, X_fix

def readDependentData():
	file = "{}/twitter_party/raw/sns.csv".format(os.path.expanduser("~"))
	df = pd.read_csv(file,delimiter=',',index_col=0,header=0)
	
    # limit it to dem/rep sample accounts and not test accounts
    df = df.loc[df['Party'].isin([0,1])]	
	
	return df
	
	
def readFollowerData():
    file = "{}/twitter_party/data/dataframe.csv".format(os.path.expanduser("~"))
    df = pd.read_csv(file,delimiter=',',index_col=0,header=0)
    return df


if __name__ == '__main__':
    
    # load in summary file with screen names
    # and merge it to file with processed follower data
    df_dependent = readDependentData()
    df_followers = readFollowerData()
    df = df_dependent.join(df_followers,how='inner')
    
    X, X_scaled, Y, scaler,X_fix = processData(df)
    #model_lr = buildLogisticModel(X_scaled,Y,X_fix,optimize=True)
    #y_probs_lr = predict(X_scaled,model_lr)
    
    # rf is slightly more accurate, but slightly lower AUC, well not really.
    model_rf = buildRandomForest(X_scaled,Y,X_fix)
    df['y_probs_rf'] = predict(X_scaled,model_rf)

    # save model to file.
    pickle.dump(model_rf,open("{}/twitter_party/data/model.model".format(os.path.expanduser("~")),"w"))

    # histogram for funsies.
    #plt.hist(y_probs_rf, 50, normed=1, facecolor='blue', alpha=0.75)
    #df['y_probs_rf'].hist(bins=50,normed=1,facecolor='blue',alpha=0.75)
    #df['y_probs_rf'].hist(by=df['Term'],sharex=True,bins=50,normed=1,facecolor='blue',alpha=0.75)

    # apply it to new pop!

