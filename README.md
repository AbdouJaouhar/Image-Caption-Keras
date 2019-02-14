# Image Captioning with Attention - Keras

An implementation of Image captioner using GRU block with Attention. It was implemented on Keras and trained on Flicker8k. More you can import clean data from Flicker descriptions files using Flickr8k.py. The model is fed with the output of the last FC layer of VGG16. You have to generate the feature.pkl file which contains predictions of VGG16 pre-trained network (cut after last FC layer) for all images in Flicker8k_Dataset.

I used the Attention Layer from these adress : https://www.kaggle.com/qqgeogor/keras-lstm-attention-glove840b-lb-0-043
