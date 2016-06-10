from setuptools import setup

setup(name='LHCPytimbermod',
	version='0.1',
	description='LHC fill data extraction tool',
	url = 'http://github.com/TMsangohan/LHCPytimbermod',
	author = 'Tom Mertens',
	author_email = 'tom.mertens@cern.ch',
	license = 'CERN',
	packages = ['LHCPytimbermod','madxmodule','PyTimber_Tom'],
	install_requires=['pandas','scipy',
		],
	include_package_data=True,
	zip_safe = False)
