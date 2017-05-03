from collectTweets import *

#######################
# This is the master script which will call functions from the other scripts.
# Will setup to get new data, scrape, model, and publish.
# then set to run on cron or remotely
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

	# build model input
	df = buildFollowerDataset(dataset='train')
	saveProcessedData(df,'dataframe')

	# build model
	df_dependent = readDependentData()
	df_followers = readFollowerData('dataframe')
	df = df_dependent.join(df_followers,how='inner')

	X, X_scaled, Y, scaler,X_fix = processData(df)
	model_rf = buildRandomForest(X_scaled,Y,X_fix)
	df['y_probs_rf'] = predict(X_scaled,model_rf)

	# save model to file.
	#pickle.dump(model_rf,open("{}/twitter_party/data/model.model".format(os.path.expanduser("~")),"w"))

	# need to build a dataset, using the same top 900 followed as in political data set.
	df_test = buildFollowerDataset(dataset='test')
	saveProcessedData(df_test,'dataframe_test')

	#load model 
	#model_rf = pickle.load(open("{}/twitter_party/data/model.model".format(os.path.expanduser("~")),"r"))

	df_test_followers = readFollowerData('dataframe_test')
	df_test = df_dependent.join(df_test_followers,how='inner')
	df_all = addrows(df,df_test)

	df_all = addrows(df,df_test)
	X, X_scaled, Y, scaler,X_fix = processData(df_all,scaler)
	df_all['y_probs_rf'] = predict(X_scaled,model_rf)
	df_new = df_all.loc[df_all['Term']=='nba']

if __name__ == "__main__":
	main()
