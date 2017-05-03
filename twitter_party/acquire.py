from collectTweets import *

#######################
# acquire!
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

	###############
	# Gather Data to Test Model
	###############
	while True:
		identifyTestAccounts(Term='nba',count=100)
		getFollowers()

if __name__ == "__main__":
	main()
