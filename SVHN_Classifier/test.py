import pdb
import tensorflow as tf
import cv2
import numpy as np
import scipy.io
import random
from sklearn.metrics import confusion_matrix, accuracy_score



mean_and_std = np.load('train_mean_and_std.npy',allow_pickle=True).item()
mean = mean_and_std['mean']
std = mean_and_std['std']


def test_on_image(image,sess,input_node,output_node):
    '''
    Function to test image using given input and output tensor node pointers and RGB converting to gray scale normalized image

    Parameters:
    ----------
        image: image in RGB 0-255 int
        sess: 'tf.Session obj' Session object with graph loaded
        input: 'tf tensor pointer' Tensor through with input is to be fed
        output: 'tf tensor pointer' Tensor through with output is to be obtained
    Returns:
    -------
        output_generated: 'nd.array' Prediction generated by model
    '''
    # Convert image to gray
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    normalized_image = (image_gray -  mean) / std
    output_generated  = sess.run(output_node, feed_dict = {input_node:np.expand_dims(normalized_image,axis=[0,3])})

    return output_generated
    

def load_graph_and_return_session(frozen_graph_file, input_node_name = 'prefix/input:0', output_node_name = 'prefix/Softmax:0' ):
    '''
    
    Parameters:
    ----------
        frozen_graph_file: 'str' file path to frozen graph
        input_node_name: 'str' Name of input node in graph
        output_node_name: 'str' Name of output node in graph
    Returns:
    --------
        sess: 'tf.Session obj' Session object with graph loaded
        input: 'tf tensor pointer' Tensor through with input is to be fed
        output: 'tf tensor pointer' Tensor through with output is to be obtained
    '''
    with tf.gfile.GFile(frozen_graph_file, "rb") as f:
        g_def = tf.GraphDef()
        g_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(g_def, name="prefix")
    
    inputs = graph.get_tensor_by_name(input_node_name)
    outputs = graph.get_tensor_by_name(output_node_name)

    sess = tf.Session(graph=graph)
    return sess, inputs, outputs

def create_frozen_graph(model_dir, model_name, output_node_names=['Softmax']):
    '''
    Function to create a frozen graph for given model

    Parameters:
    -----------
        model_dir: 'str' Dir path to which model ckpt checkpoint is stored
        model_name: 'str' Name used to save model 
        output_node_names: 'list(str)' list of node names of output from model
    Returns:
    ---------
        output_graph: 'str' Name of graph path saved     
    '''
    checkpoint_path = model_dir + model_name

    checkpoint = tf.train.get_checkpoint_state(model_dir)
    model_checkpoint = checkpoint.model_checkpoint_path

    serve_path = './served_model/{}/{}'.format(model_name,random.randint(1000, 9999))

    tf.reset_default_graph()
    
    absolute_model_dir = "/".join(model_checkpoint.split('/')[:-1])
    output_graph = absolute_model_dir + "/frozen_model.pb"

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(checkpoint_path + '.meta',clear_devices=True)
        graph = tf.get_default_graph()
        
        saver.restore(sess, model_checkpoint)
       
        '''
        ops = graph.get_operations()

        for op in ops:
            if 'Softmax' in op.name:
                print(op.name, op.values)
        '''

        output_graph_def = tf.graph_util.convert_variables_to_constants(
                        sess, # The session is used to retrieve the weights
                        tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes 
                        output_node_names # The output node names are used to select the usefull nodes
                        ) 
    
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))
        print("Output graph save path: ", output_graph)
    return output_graph


def create_some_test_images(mat_file):
    mat_data = scipy.io.loadmat(mat_file)
    images_,labels_ = mat_data['X'],mat_data['y']
    print(images_.shape, labels_.shape, "Before reshaping")
    images,labels = images_.transpose((3,0,1,2)),labels_[:,0]
    return images, labels


if __name__ == '__main__':
    # Example to create frozen graph
    model_dir = 'model_save_dir_svhn_copy/'
    model_name = 'svhn_classifier'
    graph_f = create_frozen_graph(model_dir, model_name)
    
    graph_f = 'model_save_dir_svhn_copy/frozen_model.pb'
    sess, input_n, output_n = load_graph_and_return_session(graph_f)
    
    images,labels = create_some_test_images('test_32x32.mat')
    
    gts = labels
    preds = []
    for image,label in zip(images,labels):
        pred = test_on_image(image,sess,input_n,output_n)
        pred_val = np.argmax(pred) + 1
        preds.append(pred_val)
        
    conf_mat  = confusion_matrix(gts, preds) 
    acc = accuracy_score(gts, preds)

    print("Test: Accuracy:{}, Confusion_matrix: \n".format(acc),conf_mat)

