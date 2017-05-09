from .collectTweets import *
from .buildModel import *
from .responsive import *
import pickle, bz2
import numpy as np
import pandas as pd


#######################
# setup to get new data, scrape, model, and publish.
# this is mostly a sketch outline of the steps
# just setting the responsive section to run
########################


def main():
	if not os.path.isdir('{}/twitter_party/raw/'.format(os.path.expanduser("~"))):os.makedirs('{}/twitter_party/raw'.format(os.path.expanduser("~")))

	###############
	# Gather Data to Train Model
	###############
	while False:
		# find political accounts
		identifyPoliticalAccounts(count=100)

		# for each account, make a file with a list of their followers' ids
		getFollowers()

	#############
	# Gather a test set
	#############
	while False:
		identifyTestAccounts(Term='nba',count=100)
		getFollowers()

	##############
	# Build a model
	##############3
	if False:
		df = buildFollowerDataset(dataset='train')
		#saveProcessedData(df,'dataframe')
		df_dependent = readDependentData()
		df_followers = readFollowerData('dataframe')
		df = df_dependent.join(df_followers,how='inner')
		X, X_scaled, Y, scaler,X_fix = processData(df)
		model_rf = buildRandomForest(X_scaled,Y,X_fix)
		# save model and scaler to compressed file
		#with bz2.BZ2File("{}/twitter_party/model_pickles/model.model".format(os.path.expanduser("~")),"w") as f:
		# pickle.dump(mmodel, f)

	################
	# Responsive tweeting
	################3
	if True:
		with bz2.BZ2File("{}/twitter_party/model_pickles/political.model".format(os.path.expanduser("~")),"r") as f:
			model_1 = pickle.load(f)
		with bz2.BZ2File("{}/twitter_party/model_pickles/sub1000.model".format(os.path.expanduser("~")),"r") as f:
			model_2 = pickle.load(f)
		scaler_1 = pickle.load(open("{}/twitter_party/model_pickles/political.scaler".format(os.path.expanduser("~")),"r"))
		scaler_2 = pickle.load(open("{}/twitter_party/model_pickles/sub1000.scaler".format(os.path.expanduser("~")),"r"))

		df_followers1 = readFollowerData('dataframe_political_train')
		df_followers2 = readFollowerData('dataframe_sub_train')

		followers1 = df_followers1.columns.values
		followers2 = df_followers2.columns.values
		followers1 = convertToIds(followers1.tolist())
		followers2 = convertToIds(followers2.tolist())

		fil = '{}/twitter_party/log/responses.csv'.format(os.path.expanduser("~"))
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



if __name__ == "__main__":
	main()
