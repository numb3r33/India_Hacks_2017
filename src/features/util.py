import feather
import numpy as np

def save_file(df, file_path):
	feather.write_dataframe(df, file_path)


def load_file(file_path):
	return feather.read_dataframe(file_path)


def get_feature_map(feature_name):

	if feature_name == 'genres':
		feature_map = {
			'Cricket':  '1',
			'Drama':    '2',
			'Romance':  '3',
			'Reality':  '4',
			'TalkShow': '5',
			'Comedy':   '6',
			'Family':   '7',
			'LiveTV':  '8',
			'Awards':   '2',
			'Mythology': '2',
			'Crime':     '9',
			'Action':    '10',
			'Thriller':  '11',
			'Horror':    '11',
			'Sport':     '12',
			'Football':  '12',
			'Badminton': '12',
			'Hockey':    '12',
			'Kabaddi':   '12',
			'Formula1':  '12',
			'FormulaE':  '12',
			'Table Tennis': '12',
			'Tennis'  :  '12',
			'Athletics': '12',
			'Volleyball':  '12',
			'Boxing': '12',
			'NA': '12',
			'Swimming': '12',
			'IndiaVsSa': '12',
			'Kids': '13',
			'Travel': '14',
			'Wildlife': '14',
			'Science': '15',
			'Teen': '16',
			'Documentary': '14'
		}
	
	elif feature_name == 'tod':
		feature_map = {
			'0': '1',
			'1': '2',
			'2': '3',
			'3': '4',
			'4': '5',
			'5': '6',
			'6': '7',
			'7': '8',
			'8': '9',
			'9': '10',
			'10': '11',
			'11': '12',
			'12': '13',
			'13': '14',
			'14': '15',
			'15': '16',
			'16': '17',
			'17': '18',
			'18': '19',
			'19': '20',
			'20': '21',
			'21': '22',
			'22': '23',
			'23': '24'

		}
	
	elif feature_name == 'dow':
		
		feature_map = {
			'1': '1',
			'2': '2',
			'3': '3',
			'4': '4',
			'5': '5',
			'6': '6',
			'7': '6'
		}

	else:
		feature_map = {}

	return feature_map

def remove_colons(feature):
	return feature.str.replace(r':\d+', '')

def remove_commas(feature):
	return feature.str.replace(r',', '')

def count_feature_instances(feature):
	return feature.map(lambda x: len(x.split(',')))

