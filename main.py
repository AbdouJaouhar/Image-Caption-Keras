from im2txt import Im2Txt

if __name__ == '__main__':
	ImModel = Im2Txt()
	#ImModel.Train()
	print(ImModel.GetDescription('test.jpg'))
