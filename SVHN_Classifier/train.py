import pdb
import numpy as np
import tensorflow as tf
import pdb
from sklearn.metrics import accuracy_score, confusion_matrix, balanced_accuracy_score

from data_feeder import data_feeder
from cnn_model import cnn_model

data_npy_file = 'SVHN_train_valid_splits_grayscale_colornormalized.npy'


class train_a_model():
    def __init__(self,model_config, data_file):
        self.model_config = model_config
        self.data_file = data_file
        self.epochs = self.model_config['epochs']
    
    def setup(self):
        self.data_feeeder_obj = data_feeder(self.data_file)
        self.model_obj = cnn_model(self.model_config)
        self.min_val_loss = 99999
    
    def train(self):
        model_ret = self.model_obj.get_model()

        train_to_compute = [model_ret['one_hot_output'],model_ret['loss'], model_ret['optimizer']]
        valid_to_compute = [model_ret['one_hot_output'],model_ret['loss']]
        
        total_acc = {'train':[],'valid':[]}
        total_bal_acc = {'train':[],'valid':[]}
        total_loss = {'train':[],'valid':[]}
        
        saver = tf.train.Saver()
            
        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
        

            for epoch in range(self.epochs):
                
                print(50*"*")
                print(50*"*","epoch: {}".format(epoch))
                print(50*"*")
                train_data_gen = self.data_feeeder_obj.data_generator('train')
                valid_data_gen = self.data_feeeder_obj.data_generator('valid')
                
                one_hot_preds = {'train':[],'valid':[]}
                one_hot_gts = {'train':[],'valid':[]}
                batch_loss = {'train':[],'valid':[]}
            
                for phase,generator_to_use in zip(['train','valid'],[train_data_gen, valid_data_gen]):
                    print(100*['#'])
                    print(100*['#'])
                    print("Phase:",phase)
                    print(100*['#'])
                    batch = 0
                    for image,labels in generator_to_use:
                        feed_dict = {model_ret['input']:image,\
                                        model_ret['output']:labels}

                        if phase == 'train':
                            predictions,loss,_ = sess.run(train_to_compute, feed_dict) 
                        else:
                            predictions,loss = sess.run(valid_to_compute, feed_dict)

                        one_hot_labels = [np.argmax(i) for i in labels]
                        
                        if batch == 0:
                            one_hot_preds[phase] = predictions.tolist()
                            one_hot_gts[phase] = one_hot_labels
                        else:
                            one_hot_preds[phase] += predictions.tolist()
                            one_hot_gts[phase] += one_hot_labels
                        
                        batch_loss[phase].append(loss)
                        batch = batch + 1
                    
                    print('Epoch metrics:')
                    epoch_acc = accuracy_score(one_hot_gts[phase], one_hot_preds[phase])
                    epoch_bal_acc = balanced_accuracy_score(one_hot_gts[phase], one_hot_preds[phase])
                    epoch_loss = np.mean(batch_loss[phase])
                    print("Accuracy: {}, Balanced Accuracy: {}, Loss: {}".format(epoch_acc, epoch_bal_acc, epoch_loss))
                    if phase == 'valid':
                        confusion_mat = confusion_matrix(one_hot_gts[phase], one_hot_preds[phase])
                        print('Confusion Matrix:')
                        print(confusion_mat)

                         
                    total_acc[phase].append(epoch_acc)
                    total_bal_acc[phase].append(epoch_bal_acc)
                    total_loss[phase].append(epoch_loss)

                    if phase == 'valid': 
                        if epoch_loss < self.min_val_loss:
                            self.min_val_loss = epoch_loss
                            saver.save(sess,self.model_config['model_dir'])        
                            with open('Best_confusion_matrix.txt','w') as f:
                                f.write(str(confusion_mat))

        return epoch_acc, epoch_bal_acc, epoch_loss
             

if __name__ == '__main__':
    model_config = {
            'image_dims':[32,32,1],
            'output_dims':10,\
            'keep_probability':1.0,
            'devices':['cpu:0'],
            'pos_weights': np.array([0.52764023, 0.69052669, 0.86425306, 0.98545485, 1.06943431, 1.28689065, 1.30960894, 1.45061881, 1.5673977, 1.4669587]),
            'lr':1e-4,
            'model_dir':'model_save_dir_svhn/',
            'epochs': 10
            }

    train_obj = train_a_model(model_config, data_npy_file)
    train_obj.setup()
    ret = train_obj.train()
