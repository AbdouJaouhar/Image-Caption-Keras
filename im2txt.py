from keras.layers import Conv2D,merge,Input, Dropout, Dense, LSTM, RepeatVector, Embedding, Dropout, merge, Activation,Convolution2D, MaxPooling2D, GRU,TimeDistributed, Merge
from keras.models import Sequential
from keras.layers.core import Flatten
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.models import Model
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.image import load_img
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
from numpy import array
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers.wrappers import Bidirectional

from attention import AttentionDecoder
from numpy import argmax

import numpy as np
from os import listdir
from nltk.translate.bleu_score import corpus_bleu

class Im2Txt():

	def __init__(self):
		self.directory = 'Flicker8k_Dataset'
		self.descriptions_file = 'descriptions.txt'

		self.tokenizer = Tokenizer()
		self.descriptions = dict()

		self.max_length = 0
		self.vocab_size = 4421

		self.GetDescriptions()
		self.GetIm2TxtModel()

	def preprocess_input(self,x):
	    x /= 255.
	    x -= 0.5
	    x *= 2.
	    return x

	def GetIm2TxtModel(self):

		self.image_model = Sequential()

		self.image_model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(100, 100,3)))
		self.image_model.add(Activation('relu'))
		self.image_model.add(Convolution2D(32, 3, 3))
		self.image_model.add(Activation('relu'))
		self.image_model.add(MaxPooling2D(pool_size=(2, 2)))

		self.image_model.add(Convolution2D(64, 3, 3, border_mode='valid'))
		self.image_model.add(Activation('relu'))
		self.image_model.add(Convolution2D(64, 3, 3))
		self.image_model.add(Activation('relu'))
		self.image_model.add(MaxPooling2D(pool_size=(2, 2)))

		self.image_model.add(Flatten())
		self.image_model.add(Dense(128))
		self.image_model.add(Activation('relu'))
		self.image_model.add(RepeatVector(self.max_length))

		self.language_model = Sequential()

		self.language_model.add(Embedding(self.vocab_size, 128, input_length=self.max_length))
		self.language_model.add(LSTM(256, return_sequences=True))
		self.language_model.add(AttentionDecoder(256, self.max_length))
		self.language_model.add(TimeDistributed(Dense(128)))


		self.model = Sequential()
		self.model.add(Merge([self.image_model, self.language_model], mode='concat', concat_axis=-1))
		self.model.add(Bidirectional(LSTM(256, return_sequences=False)))
		self.model.add(Dense(self.vocab_size))
		self.model.add(Activation('softmax'))

		self.model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

		print(self.model.summary())
	
	def GetData(self):

		while True:
			i = 0
			for name in listdir(self.directory):
				filename = self.directory + '/' + name

				image = load_img(filename, target_size=(100, 100))
				image = img_to_array(image)
				image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
				image = preprocess_input(image)[0]
				image_id = filename.split('/')[-1].split('.')[0]

				description = self.descriptions[image_id]

				in_img, in_seq, out_word = self.CreateSequences(description, image)
				#if i<10:
					#print(out_word)
					#i+=1
				yield [[in_img, in_seq], out_word]
			
	def GetDescriptions(self):

		file = open(self.descriptions_file, 'r')
		doc = file.read()
		file.close()

		for line in doc.split('\n')[:-1]:
			line = ''+line+''
			tokens = line.split()
			image_id, image_desc = tokens[0], tokens[1:]
			image_id = image_id.split('.')[0]

			if(image_id not in self.descriptions):
				self.descriptions[image_id] = ' '.join(image_desc)

		lines = list(self.descriptions.values())
		
		self.tokenizer.fit_on_texts(lines)
		self.max_length = max(len(s.split()) for s in list(self.descriptions.values()))
		self.vocab_size = len(self.tokenizer.word_index) + 1

	def CreateSequences(self, desc, image):

		Ximages, XSeq, y = list(), list(),list()
		
		seq = self.tokenizer.texts_to_sequences([desc])[0]

		for i in range(1, len(seq)):
			in_seq, out_seq = seq[:i], seq[i]
			in_seq = pad_sequences([in_seq], maxlen=self.max_length)[0]
			out_seq = to_categorical([out_seq], num_classes=self.vocab_size)[0]
			Ximages.append(image)
			XSeq.append(in_seq)
			y.append(out_seq)

		Ximages, XSeq, y = np.array(Ximages), np.array(XSeq), np.array(y)

		return [Ximages, XSeq, y]


	def Train(self):
		model_checkpoint = ModelCheckpoint('weights.{epoch:02d}-{loss:.2f}.hdf5', monitor='loss', verbose=0,
		                                  save_best_only=True, mode='min')

		tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=False)
		callbacks_list = [model_checkpoint, tensorboard]

		early_stopping = EarlyStopping(monitor='loss', patience=2)
		self.model.fit_generator(self.GetData(), steps_per_epoch=50, callbacks=callbacks_list, epochs=20)
		self.model.save_weights('Im2Txt.h5')

	def GetFeatures(self,filename):

		image = load_img(filename, target_size=(100,100))
		image = img_to_array(image)
		image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
		image = preprocess_input(image)

		return image


	def GetWord(self,integer):
		for word, index in self.tokenizer.word_index.items():
			if index == integer:
				return word

		return None
	 
	def GetDescription(self,photo):
		in_text = '<start>'

		for i in range(self.max_length):
			sequence = self.tokenizer.texts_to_sequences([in_text])[0]
			sequence = pad_sequences([sequence], maxlen=self.max_length)
			yhat = self.model.predict([self.GetFeatures(photo),sequence], verbose=0)
			yhat = argmax(yhat)
			word = self.GetWord(yhat)

			if word is None:
				in_text += '<end>'
				break
			in_text += ' ' + word

			if word == '<end>':
				in_text += '<end>'
				break

		print(in_text,len(in_text.split()))
		return in_text
