import tensorflow as tf

class cnn_model(object):
    def __init__(self, model_config):
        '''
        model_config: 'dict'
                        'image_dims': [batchsize,width,height,channels] 
        '''
        self.model_config = model_config
        self.image_dims = self.model_config['image_dims']
        self.output_dims = self.model_config['output_dims']
        self.keep_probability = self.model_config['keep_probability']
        self.pos_weights = self.model_config['pos_weights'] 
        self.input = tf.placeholder(tf.float32,[None,self.image_dims[0],self.image_dims[1],self.image_dims[2]],name='input')
        self.output = tf.placeholder(tf.float32,[None, self.output_dims],name='output')

        self.initializer = tf.contrib.layers.xavier_initializer()
    
    def create_weights_(self, name, shape, initializer=tf.contrib.layers.xavier_initializer()):
        return tf.get_variable(name=name, shape=shape, initializer=initializer)

    def create_biases_(self, name, shape, initializer=tf.constant_initializer(0.01)):
        return tf.get_variable(name=name, shape=shape, initializer=initializer)

    def conv_2d(self, inp, weights, strides=1):
        return tf.nn.conv2d(inp, weights, strides=[1, strides, strides, 1], padding="SAME")

    def max_pool_2d(self, inp, kernel=3, strides=2):
        return tf.nn.max_pool(inp, ksize=[1, kernel, kernel, 1], strides=[1, strides, strides, 1], padding="SAME")

    def conv_max_pool(self, inp, w, b):
        conv = tf.nn.relu(tf.nn.bias_add(self.conv_2d(inp, w), b))
        return self.max_pool_2d(conv)

    def dense_layer(self, inp, w, b, activation=True, keep_prob = 1.0):
        if activation:
            return tf.nn.dropout(tf.nn.relu(tf.matmul(inp, w) + b), keep_prob)
        else:
            return tf.nn.dropout(tf.matmul(inp, w) + b, keep_prob)


    def get_model(self):
        with tf.device(self.model_config['devices'][0]):
            with tf.name_scope("conv_1"):
                conv_1_w = self.create_weights_("conv_1w",shape=[3,3,1,32])
                conv_1_b = self.create_biases_("conv_1b",shape=[32])
            with tf.name_scope("conv_2"):
                conv_2_w = self.create_weights_("conv_2w",shape=[3,3,32,64])
                conv_2_b = self.create_biases_("conv_2b",shape=[64])
            with tf.name_scope("conv_3"):
                conv_3_w = self.create_weights_("conv_3w",shape=[3,3,64,128])
                conv_3_b = self.create_biases_("conv_3b",shape=[128])
            with tf.name_scope("full1"):
                full_w = self.create_weights_("fullw",shape=[4*4*128,256])
                full_b = self.create_biases_("fullb",shape = [256])
            with tf.name_scope('full_logits'):
                full_logits_w = self.create_weights_("fulllogitw",shape=[256,self.output_dims])
                full_logits_b = self.create_biases_("fulllogitb",shape = [self.output_dims])
        
        with tf.device(self.model_config['devices'][0]):
            conv1_out = self.conv_max_pool(self.input, conv_1_w, conv_1_b)
            conv2_out = self.conv_max_pool(conv1_out, conv_2_w, conv_2_b)
            conv3_out = self.conv_max_pool(conv2_out, conv_3_w, conv_3_b)
            
            flat_conv3_out = tf.reshape(conv3_out,[-1,4*4*128])
            
            full_out = self.dense_layer(flat_conv3_out,full_w,full_b,activation=True,keep_prob=0.3)
            
            softmax_logits_out = tf.nn.softmax(self.dense_layer(full_out,full_logits_w,full_logits_b,activation=True,keep_prob=1.0),axis=1)
            one_hot_out = tf.math.argmax(softmax_logits_out,axis=1)
            
        with tf.name_scope('loss'):
            logits_ = softmax_logits_out
            targets_ = self.output
            loss = tf.reduce_mean(
                    tf.nn.weighted_cross_entropy_with_logits(targets=targets_, logits=logits_, pos_weight=self.pos_weights)
                    )

        with tf.name_scope('optimizer'):
             optimizer = tf.train.AdamOptimizer(self.model_config["lr"]).minimize(loss)
 

        return {'softmax_logits_out':softmax_logits_out,'input':self.input,'output':self.output, 'loss':loss,
                'optimizer':optimizer,'one_hot_output':one_hot_out}


if __name__ == '__main__':
    # Debug statements to build the model
    import numpy as np

    dummy_input_batch = np.zeros([2,32,32,1])
    
    dummy_out = 10*[0]
    dummy_out[0] = 1
    dummy_output_batch = []
    dummy_output_batch.append(dummy_out)
    dummy_out = 10*[0]
    dummy_out[0] = 4
    dummy_output_batch = []
    dummy_output_batch.append(dummy_out)


    model_config = {'image_dims':[32,32,1],
                    'output_dims':10,\
                    'keep_probability':1.0,
                    'devices':['cpu:0'],
                    'pos_weights': np.array(10*[1.]),
                    'lr':1e-4}
    
    model_obj = cnn_model(model_config)
    
    model_ret = model_obj.get_model()
    
    init_op = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init_op)
        to_compute = [model_ret['loss'],model_ret['softmax_logits_out']] # What are the value to be computed
        feed_dict = {model_ret['input']:dummy_input_batch,
                model_ret['output']:dummy_output_batch} # Input value to be fed association
        output = sess.run(to_compute, feed_dict)
