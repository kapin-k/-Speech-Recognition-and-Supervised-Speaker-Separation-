import os
from sklearn.model_selection import train_test_split
if not os.path.exists('training_data'):
	os.makedirs("./training_data")
if not os.path.exists('testing_data'):
	os.makedirs("./testing_data")
for root, dirs, files in os.walk("./spectrogram_images", topdown=False):
	if root == "./spectrogram_images":
		for name in dirs:
			X =y = os.listdir(root+"/"+name)
			if not os.path.exists("./training_data/"+name):
				os.makedirs("./training_data/"+name)
			if not os.path.exists("./testing_data/"+name):
				os.makedirs("./testing_data/"+name)
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)#test_size=0.1
			for x in X_train:
				print(root+"/"+name+"/"+x,"./training_data/"+name+"/"+x)
				os.rename(root+"/"+name+"/"+x,  "./training_data/"+name+"/"+x)
			for x in X_test:
				print(root+"/"+name+"/"+x,"./testing_data/"+name+"/"+x)
				os.rename(root+"/"+name+"/"+x, "./testing_data/"+name+"/"+x)
