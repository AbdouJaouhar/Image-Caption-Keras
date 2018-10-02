# Image Captioning with Attention - Keras

An implementation of Image captioner using GRU block with Attention. It was implemented on Keras and trained on Flicker8k. More you can find a function in Flickr8k.py to import clean data from Flicker descriptions files. The model is feed with the output of last FC layer of VGG16. You have to generate the feature.pkl file wich contain prediction of VGG16 pretrained network (cut after last FC layer) for all images of Flicker8k_Dataset.
