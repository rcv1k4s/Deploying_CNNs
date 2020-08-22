import numpy as np 
import math

class data_feeder():
    def __init__(self, data_file, batch_size = 40):
        data = np.load(data_file,allow_pickle=True).item()
        
        self.batch_size = batch_size

        self.train_images = data['train_images_grayscale']
        self.train_labels = data['train_one_hot_labels']
        
        self.valid_images = data['valid_images_grayscale']
        self.valid_labels = data['valid_one_hot_labels']

        self.train_len = len(self.train_labels)
        self.valid_len = len(self.valid_labels)
        
        self.train_minibatches = int(math.ceil(self.train_len/float(self.batch_size)))
        self.valid_minibatches = int(math.ceil(self.valid_len/float(self.batch_size)))

    def data_generator(self,phase):
        if phase == 'train':
            no_of_minibatches = self.train_minibatches
            images = self.train_images
            labels = self.train_labels
        elif phase == 'valid':
            no_of_minibatches = self.valid_minibatches
            images = self.valid_images
            labels = self.valid_labels
        else:
            raise 'Unknown phase only train and valid are accepted!'
        
        for split_index in range(no_of_minibatches):
            from_split_index = split_index * self.batch_size
            to_split_index = (split_index + 1) * self.batch_size
            current_images = images[from_split_index:to_split_index] 
            current_labels = labels[from_split_index:to_split_index]
            yield np.expand_dims(current_images,axis=3), current_labels


if __name__ == '__main__':
    # Test the data feeder
    data_file = 'SVHN_train_valid_splits_grayscale_colornormalized.npy'
    data_feeder_obj = data_feeder(data_file)

    # train_generator
    train_gen = data_feeder_obj.data_generator('train')
    # valid_generator
    valid_gen = data_feeder_obj.data_generator('valid')


