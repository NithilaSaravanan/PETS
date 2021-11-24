#Plugging in an empty script here - to be filled in 
import argparse
import sys
import sys
import os.path as osp

this_dir = osp.dirname(__file__)

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('-c', help = 'Enter the configuration file with all the tunable parameters for the method selected', default = 'nope')
	parser.add_argument('-d', help = 'Enter the data file with the values of the signal (Y) to be filtered', default = 'nope')
	parser.add_argument('-m', help = 'Select the estimation method you want to use (ADD VALS HERE)', default = 'nope')
	parser.add_argument('-g',help = 'Select whether you want graphs in the output directory (Y / N)',default = 'Y')
	try:
		args =parser.parse_args()
	except:
		print('\n One or more parameters are missing or incorrect. Please take a look at the help and provide correct arguments')
    	parser.print_help()
    	sys.exit(0)

    #Storing arguments in variables to be used later
	config_file = args.c
	data_file = args.d
	method_select = args.m
	graph_choice = args.g

    #Checking if important arguments were provided or not
	if config_file == 'nope':
		print('\n Please provide the config file. You can paste the file path after -c \n')
		parser.print_help()
		sys.exit()

	if data_file == 'nope':
		print('\n Please provide the data file. You can paste the file path after -d \n')
		parser.print_help()
		sys.exit()

	if method_select == 'nope':
		print('\n Please select one of the methods of estimation and try again. \n')
		parser.print_help()
		sys.exit()

if __name__=='__main__':
	main()
