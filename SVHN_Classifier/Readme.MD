
Dataset Links:
	http://ufldl.stanford.edu/housenumbers/train_32x32.mat
	http://ufldl.stanford.edu/housenumbers/test_32x32.mat
	http://ufldl.stanford.edu/housenumbers/extra_32x32.mat

Download the dataset from above links 

process_data.py script to process the data files download to create:
	1. Convert data to grayscale
	2. Normalize using dataset mean and standard deviation
	3. Create train and valid dataset splits
	4. Compute loss weights
	5. Create a npy file with all the required data

cnn_model.py: Tensorflow model to be used to train

data_feeder.py Module to dynamically yield images, labels through generator to be fed to model in tensorflow session

train.py Script to train the model using above modules and log per epoch results



