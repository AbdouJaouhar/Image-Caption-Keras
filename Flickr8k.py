import pandas as pd 
from pickle import load
from keras.preprocessing.text import Tokenizer

def Flickr8k(options):

	# Load descriptions files
	PandaFile = pd.read_csv(options['DescriptionFile'], sep="\t", names=['image_id', 'description'])
	PandaFileTrain = pd.read_csv(options['TrainFile'], sep="\t", names=['image_id'])
	PandaFileTest = pd.read_csv(options['TestFile'], sep="\t", names=['image_id'])

	# Definition Operations for processing datas: delete '.jpg' from image_id and add <start>/<end> to descriptions
	DeleteJPGExtension = lambda x: x.split('.jpg')[0]
	AddStartEndDelimiter = lambda x: '<start> ' + x + ' <end>'

	# Apply operations to datas
	PandaFile['image_id'] = PandaFile['image_id'].apply(DeleteJPGExtension)
	PandaFileTrain['image_id'] = PandaFileTrain['image_id'].apply(DeleteJPGExtension)
	PandaFileTest['image_id'] = PandaFileTest['image_id'].apply(DeleteJPGExtension)

	PandaFile['description'] = PandaFile['description'].apply(AddStartEndDelimiter)

	# Get descriptions for Train and Test image_id
	TrainDatas = PandaFile.merge(PandaFileTrain, how='inner', on=['image_id']).drop_duplicates(['image_id']).reset_index()
	TestDatas = PandaFile.merge(PandaFileTest, how='inner', on=['image_id']).drop_duplicates(['image_id']).reset_index()
	
	# Tokenize DescriptionFile to get vocabulary size and maximum-length of a description
	FileTokenizer = Tokenizer()
	FileTokenizer.fit_on_texts(PandaFile['description'].tolist())

	VocabularySize = len(FileTokenizer.word_index) + 1
	MaxLength = PandaFile['description'].map(lambda x: len(x.split(' '))).max()

	AllFeatures = load(open('features.pkl', 'rb'))
	TrainFeatures = {k: AllFeatures[k] for k in list(TrainDatas['image_id'])}
	TestFeatures = {k: AllFeatures[k] for k in list(TestDatas['image_id'])}

	return {'FileTokenizer' : FileTokenizer, 'TestFeatures' : TestFeatures, 'TrainFeatures' : TrainFeatures, 'TrainDatas' : TrainDatas, 'TestDatas' : TestDatas, 'VocabularySize' : VocabularySize, 'MaxLength' : MaxLength}
