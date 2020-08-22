import scipy.io
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from random import shuffle, seed

def convert_image_to_gray(image):
    return cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

mat_file_name = 'train_32x32.mat'

def read_mat_file(mat_file_name):
    '''
    Function loads SVHN mat file to give out gray images and labels
    Return:
    --------
            Images in gray scale 32x32x1
            labels in one hot vector 1 - 10 mapped to one hot vector of 10 values
                0 - [1,0,0,0,0,0,0,0,0,0]
                1 - [0,1,0,0,0,0,0,0,0,0]
                10 - [0,0,0,0,0,0,0,0,0,10]
    '''
    ret = scipy.io.loadmat(mat_file_name)
    images_,labels_ = ret['X'],ret['y']
    print(images_.shape, labels_.shape, "Before reshaping")
    images,labels = images_.transpose((3,0,1,2)),labels_[:,0]
    print(images.shape, labels.shape, "Before reshaping")
    gray_images = [convert_image_to_gray(image) for image in images]
    
    one_hot_labels = []

    for label in labels:
        onehot_ref = 10*[0]
        onehot_ref[label -1] = 1
        one_hot_labels.append(onehot_ref)
    
    seed(41)
    shuffle(one_hot_labels)

    seed(41)
    shuffle(gray_images)


    images_train, images_val, labels_train, labels_val = train_test_split(gray_images, one_hot_labels, test_size=0.20, random_state=10)
    
    train_mean,train_std = find_train_mean_and_standard_deviation(images_train)
    
    # Substract mean and std deviation
    train_greyscale_norm = (images_train - train_mean) / train_std
    test_greyscale_norm = (images_val - train_mean)  / train_std

    return train_greyscale_norm,test_greyscale_norm,labels_train, labels_val, train_mean, train_std

def find_train_mean_and_standard_deviation(train_images):
    return np.mean(train_images,axis=0), np.std(train_images,axis=0)

if __name__ == '__main__':
    train_images,valid_images,train_labels,valid_labels,train_mean,train_std = read_mat_file(mat_file_name)
    lab = [np.argmax(i) for i in train_labels]
    class_weights = compute_class_weight('balanced',list(range(10)),lab)
    print(class_weights)
    save_dict = {'train_images_grayscale':train_images,\
            'valid_images_grayscale':valid_images,\
            'train_one_hot_labels':train_labels,\
            'valid_one_hot_labels':valid_labels,\
            'train_mean': train_mean,\
            'train_std': train_std,\
            'class_weights':class_weights,\
            }
    np.save('SVHN_train_valid_splits_grayscale_colornormalized',save_dict)
