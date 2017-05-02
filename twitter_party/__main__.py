from collectTweets import *

#######################
# This is the master script which will call functions from the other scripts.
# Will setup to get new data, scrape, model, and publish.
# then set to run on cron or remotely
########################


def main():
	if not os.path.isdir('{}/twitter_party/raw/'.format(os.path.expanduser("~"))):os.makedirs('{}/twitter_party/raw'.format(os.path.expanduser("~")))

	###############
	# Gather Data
	###############
	while False:
		#identifyPoliticalAccounts(count=100)

		# for each account, make a file with a list of their followers' ids
		getFollowers()


	# test set
	identifyTestAccounts(Term='nba',count=100)
	getFollowers()

if __name__ == "__main__":
	main()
