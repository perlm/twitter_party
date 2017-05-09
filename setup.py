#!/usr/bin/python

from distutils.core import setup

setup(name='twitter_party',
	version='0.1.0',
	description="Twitter Party prediction model", 
	packages=['twitter_party'],
	install_requires=['numpy','pandas','sklearn','twitter'],
	entry_points={
		'console_scripts': [
		'twitter_party = twitter_party.__main__:main'
		]
	}
	)
