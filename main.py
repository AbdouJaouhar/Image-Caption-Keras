from im2txt import Im2Txt

if __name__ == '__main__':
	ImModel = Im2Txt()
	#ImModel.Train()
	ImModel.GenDescription(ImModel.extract_features('test.jpg'))
