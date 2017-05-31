import feather

def save_file(df, file_path):
	feather.write_dataframe(df, file_path)


def load_file(file_path):
	return feather.read_dataframe(file_path)