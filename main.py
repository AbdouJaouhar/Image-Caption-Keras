from ImageCaptioning import ImageCaptioning
from Flickr8k import Flickr8k

flickr_options = {'DatasetFolder' : 'Flicker8k_Dataset', 
				  'DescriptionFile' : 'Flickr8k_text/Flickr8k.token.txt', 
				  'TrainFile' : 'Flickr8k_text/Flickr_8k.trainImages.txt', 
				  'TestFile' : 'Flickr8k_text/Flickr_8k.testImages.txt'}

FlickrDatas = Flickr8k(flickr_options)

ImageCaptioning = ImageCaptioning(FlickrDatas)
ImageCaptioning.build()
# ImageCaptioning.train(30)
ImageCaptioning.LoadModel('model_29.h5')

description = ImageCaptioning.GetGenerateDescription('Flicker8k_Dataset/237547381_aa17c805e0.jpg')

print("Description generated : ", description)
