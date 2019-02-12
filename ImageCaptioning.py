import numpy as np 
from keras.applications.vgg16 import VGG16
import time
from keras.models import Model
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.applications.vgg16 import preprocess_input
from os import listdir
from keras.layers import Input, add
from keras.layers import Dense, Concatenate
from keras.layers import LSTM, Bidirectional, GRU
from keras.layers import Embedding
from keras.layers import Dropout
from Attention import Attention

from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

class ImageCaptioning():

	def __init__(self, FlickrDatas):
		self.max_length = FlickrDatas['MaxLength']
		self.vocab_size = FlickrDatas['VocabularySize']
		self.TestDatas = FlickrDatas['TestDatas']
		self.TrainDatas = FlickrDatas['TrainDatas']
		self.TestFeatures = FlickrDatas['TestFeatures']
		self.TrainFeatures = FlickrDatas['TrainFeatures']
		self.FileTokenizer = FlickrDatas['FileTokenizer']
		self.model = None


	def build(self):
		# Image Encoder
		ImageEncoderInput = Input(shape=(4096,))
		ImageEncoder = Dropout(0.35)(ImageEncoderInput)
		ImageEncoder = Dense(256, activation='relu')(ImageEncoder)

		# Language Encoder
		LanguageEncoderInput = Input(shape=(self.max_length,))
		LanguageEncoder = Embedding(self.vocab_size, 128, mask_zero=True)(LanguageEncoderInput)
		LanguageEncoder = Dropout(0.35)(LanguageEncoder)
		LanguageEncoder = Bidirectional(GRU(128, return_sequences=True, dropout=0.25,recurrent_dropout=0.25))(LanguageEncoder) 
		LanguageEncoder = Attention(self.max_length)(LanguageEncoder)

		# Decoder
		Decoder = add([ImageEncoder, LanguageEncoder])
		Decoder = Dense(500, activation='relu')(Decoder)
		Decoder = Dense(self.vocab_size, activation='softmax')(Decoder)

		self.model = Model(inputs=[ImageEncoderInput, LanguageEncoderInput], outputs=Decoder)
		self.model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-3,decay=1e-5))
		self.model.summary()
	
	def train(self, epochs):
		for i in range(epochs):
			generator = self.Generator()
			self.model.fit_generator(generator, epochs=1, steps_per_epoch=len(self.TrainDatas), verbose=1)
			self.model.save('model_' + str(i) + '.h5')

	def Generator(self):
		while True:
			for i in range(len(self.TrainDatas)):
				image_feature = self.TrainFeatures[self.TrainDatas['image_id'][i]][0]
				image_description = self.TrainDatas['description'][i]

				in_img ,in_seq ,out_word = self.GetSequences(image_feature, image_description)
				yield [[in_img, in_seq], out_word]

	def GetSequences(self, image_feature, image_description):
		Images, InputDescriptions, OutputWords = list(), list(), list()

		SequenceDescription = self.FileTokenizer.texts_to_sequences([image_description])[0]
		for i in range(1, len(SequenceDescription)):
			InputDescription = pad_sequences([SequenceDescription[:i]], maxlen=self.max_length)[0]
			OutputWord = to_categorical([SequenceDescription[i]], num_classes=self.vocab_size)[0]
			Images.append(image_feature)
			InputDescriptions.append(InputDescription)
			OutputWords.append(OutputWord)

		return np.array(Images), np.array(InputDescriptions), np.array(OutputWords)

	def GetFeatures(self, filename):
		model = VGG16()
		model.layers.pop()
		model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
		image = load_img(filename, target_size=(224, 224))
		image = img_to_array(image)
		image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
		image = preprocess_input(image)
		feature = model.predict(image, verbose=0)

		return feature

	def LoadModel(self,model_weights):
		self.model.load_weights(model_weights)

	def GetGenerateDescription(self,photo):
		photo = self.GetFeatures(photo)

		description = '<start>'
		for i in range(self.max_length):
			sequence = self.FileTokenizer.texts_to_sequences([description])[0]
			sequence = pad_sequences([sequence], maxlen=self.max_length)
			yhat = self.model.predict([photo,sequence], verbose=0)
			yhat = np.argmax(yhat)
			word = [x for x,y in list(self.FileTokenizer.word_index.items()) if y == yhat]
			if word is []:
				break
			description += ' ' + word[0]
			if word == '<end>':
				break
		print("\n",)
		print(description)
		return description
