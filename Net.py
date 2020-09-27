import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class Net:

    def __init__(self, args, print_and_log, save_figure, optimizer_obj, optimizer_args):

        self.args          = args
        self.print_and_log = print_and_log
        self.save_figure   = save_figure
        self.graph         = tf.Graph()

        with self.graph.as_default():
            # Inputs
            self.x             = tf.compat.v1.placeholder(name='input',         dtype=tf.float32, shape=(None,) + args['input_shape'])
            self.y             = tf.compat.v1.placeholder(name='target',        dtype=tf.float32, shape=(None,    args['num_classes']))
            self.learning_rate = tf.compat.v1.placeholder(name='learning_rate', dtype=tf.float32, shape=())
            self.is_train      = tf.compat.v1.placeholder_with_default(False,   name='is_train',  shape=[])

            # Model
            self.output = self.build_network()
            self.count_free_parameters()

            # Loss
            self.accuracy   = self.compute_accuracy()
            self.jacob_loss = self.compute_jacobian_loss()
            self.l2_loss    = tf.reduce_sum(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES))
            self.cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                     labels = self.y,
                     logits = self.output,
                     name   = 'softmax_cross_entropy_loss'))
            self.entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                     labels = tf.nn.softmax(self.output),
                     logits = self.output,
                     name   = 'entropy_loss'))
            loss = self.cross_entropy_loss + self.l2_loss + args['entropy_coef']*self.entropy_loss

            # Optimizer
            optimizer           = optimizer_obj(self.learning_rate, **optimizer_args).minimize(loss) 
            optimizer_jacob     = optimizer_obj(self.learning_rate, **optimizer_args).minimize(loss + args['jacob_coef']*tf.reduce_mean(self.jacob_loss))
            update_ops          = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
            self.train_op       = tf.group([optimizer,       update_ops]) # necessary for batchnorm
            self.train_op_jacob = tf.group([optimizer_jacob, update_ops]) # necessary for batchnorm          
                        
            # General ops
            self.init  = tf.compat.v1.global_variables_initializer()
            self.saver = tf.compat.v1.train.Saver(max_to_keep=None)

        self.session = tf.compat.v1.Session(graph=self.graph)

    def build_network(self):
        raise NotImplementedError
    
    def conv_layer(self, name, layer_input, kernel_size, out_channels, stride=1, with_relu=True, padding='SAME', with_batchnorm=True):
        """
        A block of layers:
        convolutional
        batch normalization
        ReLU
        """
        W = tf.get_variable(name=name + '_weights',
                            shape=[kernel_size, kernel_size, layer_input.get_shape()[-1], out_channels],
                            initializer=tf.contrib.layers.variance_scaling_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(self.args['weight_decay_coef']))
        if self.args['do_spectral_norm']:
            u = tf.get_variable(name + '_u', [1, out_channels], initializer=tf.random_normal_initializer(), trainable=False)
            W = self.spectral_normalize(W, u)

        conv_lyr = tf.nn.conv2d(input=layer_input, filter=W, strides=[1, stride, stride, 1], padding=padding, name=name)

        if not with_relu:
            bias = tf.get_variable(name=name + '_bias', shape=[out_channels], initializer=tf.constant_initializer(0.0))
            conv_lyr = tf.nn.bias_add(conv_lyr, bias, name=name + '_biased')

        if with_batchnorm:
            conv_lyr = tf.layers.batch_normalization(
                inputs            = conv_lyr,
                training          = self.is_train,
                scale             = not self.args['do_spectral_norm'],
                beta_initializer  = tf.constant_initializer(0.1),
                gamma_regularizer = tf.contrib.layers.l2_regularizer(self.args['weight_decay_coef']),
                name              = name + '_batchnorm')

        if with_relu:
            conv_lyr = tf.nn.relu(conv_lyr, name=name + '_relu')

        return conv_lyr

    def dense_layer(self, name, layer_input, output_size, with_batchnorm_and_relu=True):
        """
        A block of layers:
        fully conected
        batch normalization
        ReLU
        """
        input_shape = layer_input.shape.as_list()
        W = tf.get_variable(name=name+'_weights', shape=[input_shape[-1], output_size],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(self.args['weight_decay_coef']))
        if self.args['do_spectral_norm']:
            u = tf.get_variable(name+'_u', [1, output_size], initializer=tf.random_normal_initializer(), trainable=False)
            W = self.spectral_normalize(W, u)

        dense_lyr = tf.matmul(layer_input, W)

        if with_batchnorm_and_relu:
            dense_lyr = tf.layers.batch_normalization(
                inputs            = dense_lyr,
                training          = self.is_train,
                scale             = not self.args['do_spectral_norm'],
                beta_initializer  = tf.constant_initializer(0.1),
                gamma_regularizer = tf.contrib.layers.l2_regularizer(self.args['weight_decay_coef']),
                name              = name + '_batchnorm')
            dense_lyr = tf.nn.relu(dense_lyr, name=name + '_relu')
        else:
            bias      = tf.get_variable(name=name+'_bias', shape=[output_size], initializer=tf.zeros_initializer())
            dense_lyr = tf.nn.bias_add(dense_lyr, bias, name=name + '_biased')
       
        return dense_lyr
    
    @staticmethod
    def spectral_normalize(W_, u_, iterations=1):
        W_shape = W_.shape.as_list()
        W       = tf.reshape(W_, [-1, W_shape[-1]])  # Each 3D output kernel -> column
        u       = u_
        for i in range(iterations):
            """ power iteration (usually 1 is enough) """
            v = tf.nn.l2_normalize(tf.matmul(u, tf.transpose(W)))
            u = tf.nn.l2_normalize(tf.matmul(v, W))
        sigma = tf.matmul(tf.matmul(v, W), tf.transpose(u))
        with tf.control_dependencies([u_.assign(u)]):
            W_normalize = W_ / sigma
        return W_normalize

    def compute_accuracy(self):
        predict_per_sample    = tf.argmax(self.output, axis=1)
        label_per_sample      = tf.argmax(self.y,      axis=1)
        is_correct_per_sample = tf.cast(tf.equal(predict_per_sample, label_per_sample), tf.float32)
        return tf.reduce_mean(is_correct_per_sample) * 100

    def compute_jacobian_loss(self):
        gradients = tf.gradients(self.output * tf.random_normal((tf.shape(self.output))), self.x)[0]
        return tf.reduce_sum(gradients**2, axis=(1,2,3)) # Squared Frobenious norm per sample

    def train(self, args):

        start_t = time.time()
        self.print_and_log('\nStart training - ' + time.asctime(time.localtime(start_t)))

        # initialize variables, either from checkpoint or randomly:        
        if args['restore_initalization']:
            self.print_and_log('restoring initialization')            
            self.saver.restore(self.session, args['restore_from_path'])
        else:
            self.session.run(self.init)
            
        history = {
            'train_true_loss'      : [],
            'train_noisy_loss'     : [],
            'valid_loss'           : [],
            'test_loss'            : [],
            'train_true_accuracy'  : [],
            'train_noisy_accuracy' : [],
            'valid_accuracy'       : [],
            'test_accuracy'        : []}

        train_size    = args['train_data'].shape[0]
        index         = np.arange(train_size)
        best_accuracy = 0.

        for epoch in range(args['epochs_num']):
            self.print_and_log('\n*** Epoch %d/%d ***' % (epoch + 1, args['epochs_num']))
            np.random.shuffle(index)
            images        = args['train_data'][index, ...]
            labels        = args['train_target'][index, ...]
            learning_rate = args['initial_lr'] * (args['decrease_lr_factor']) ** np.floor(epoch / args['decrease_lr_every'])            
            self.print_and_log('lr: ' + str(learning_rate))
            for batch_num in range(int(np.ceil(train_size / args['train_batch_size']))):
                batch_start = batch_num   * args['train_batch_size']
                batch_end   = batch_start + args['train_batch_size']
                if batch_end > train_size:
                    batch_end = train_size
                batch_data   = images[batch_start : batch_end, ...]
                batch_target = labels[batch_start : batch_end, ...]
                feed_dict = {self.x : batch_data, self.y : batch_target, self.is_train : True, self.learning_rate : learning_rate}                
                if self.args['jacob_coef'] and epoch >= 10:
                    self.session.run([self.train_op_jacob], feed_dict=feed_dict) 
                    if not batch_num:
                        self.print_and_log('jacob coef: ' + str(self.args['jacob_coef']))   
                else:
                    self.session.run([self.train_op], feed_dict=feed_dict)                  
                    
            # Evaluation
            valid_loss,       valid_accuracy       = self.evaluate_prediction(args['valid_data'], args['valid_target'],      args['valid_batch_size'])
            test_loss,        test_accuracy        = self.evaluate_prediction(args['test_data'],  args['test_target'],       args['valid_batch_size'])
            train_noisy_loss, train_noisy_accuracy = self.evaluate_prediction(args['train_data'], args['train_target'],      args['valid_batch_size'])
            train_true_loss,  train_true_accuracy  = self.evaluate_prediction(args['train_data'], args['train_true_target'], args['valid_batch_size'])

            history['train_true_loss']     .append(train_true_loss)
            history['train_noisy_loss']    .append(train_noisy_loss)
            history['valid_loss']          .append(valid_loss)
            history['test_loss']           .append(test_loss)
            history['train_true_accuracy'] .append(train_true_accuracy)
            history['train_noisy_accuracy'].append(train_noisy_accuracy)
            history['valid_accuracy']      .append(valid_accuracy)
            history['test_accuracy']       .append(test_accuracy)

            self.print_and_log('Train noisy loss: %s, accuracy: %f' % (str(train_noisy_loss), train_noisy_accuracy))
            self.print_and_log('Train true loss:  %s, accuracy: %f' % (str(train_true_loss),  train_true_accuracy))
            self.print_and_log('Valid loss:       %s, accuracy: %f' % (str(valid_loss),       valid_accuracy))
            self.print_and_log('Test loss:        %s, accuracy: %f' % (str(test_loss),        test_accuracy))

            if valid_accuracy > best_accuracy:
                best_accuracy = valid_accuracy
                self.saver.save(self.session, args['save_to_path'] + '_epoch_' + str(epoch + 1))
                self.print_and_log('Saving checkpoint, epoch %d' % (epoch + 1))

        plt.plot(range(1, args['epochs_num'] + 1), history['train_noisy_loss'])
        plt.plot(range(1, args['epochs_num'] + 1), history['train_true_loss'])
        plt.plot(range(1, args['epochs_num'] + 1), history['valid_loss'])
        plt.plot(range(1, args['epochs_num'] + 1), history['test_loss'])
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss per epoch')
        plt.legend(('Train noisy', 'Train true', 'Valid', 'Test'))
        self.save_figure('loss_')
        plt.show(block=False)
        plt.close()

        plt.plot(range(1, args['epochs_num'] + 1), history['train_noisy_accuracy'])
        plt.plot(range(1, args['epochs_num'] + 1), history['train_true_accuracy'])
        plt.plot(range(1, args['epochs_num'] + 1), history['valid_accuracy'])
        plt.plot(range(1, args['epochs_num'] + 1), history['test_accuracy'])
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy per epoch')
        plt.legend(('Train noisy', 'Train true', 'Valid', 'Test'))
        self.save_figure('accuracy_')
        plt.show(block=False)
        plt.close()

        end_t = time.time()
        self.print_and_log('\nEnd training -' + time.asctime(time.localtime(end_t)))
        self.print_and_log('Total time: %f minutes' % ((end_t - start_t) / 60))

        return history

    def evaluate_prediction(self, data, target, batch_size=32):
        data_size    = data.shape[0]
        factor       = batch_size / data_size
        avg_loss     = 0.
        avg_accuracy = 0.
        for batch_num in range(int(np.ceil(data_size / batch_size))):
            batch_start = batch_num * batch_size
            batch_end   = batch_start + batch_size
            if batch_end > data_size:
                batch_end = data_size
                factor    = (batch_end - batch_start) / data_size
            feed_dict = {self.x: data[batch_start: batch_end, ...], self.y: target[batch_start: batch_end, ...]}
            batch_loss, batch_accuracy = self.session.run([self.cross_entropy_loss, self.accuracy], feed_dict=feed_dict)
            avg_loss += batch_loss * factor
            avg_accuracy += batch_accuracy * factor
        return avg_loss, avg_accuracy

    def predict(self, data, batch_size=32):
        data_size = data.shape[0]
        logits    = np.zeros((data_size, self.args['num_classes']), dtype=np.float32)
        for batch_num in range(int(np.ceil(data_size / batch_size))):
            batch_start  = batch_num * batch_size
            batch_end    = batch_start + batch_size
            if batch_end > data_size: batch_end = data_size
            feed_dict    = {self.x: data[batch_start: batch_end, ...]}
            logits[batch_start: batch_end] = self.session.run(self.output, feed_dict=feed_dict)
        return logits

    def load_weights(self, file_path):
        self.saver.restore(self.session, file_path)

    def count_free_parameters(self):
        """
        count and report free paramwters of the net
        """
        self.print_and_log('\n** Network free parameters **\n')
        total_free_parameters = 0
        for tensor in tf.compat.v1.trainable_variables():
            free_parameters = np.prod(np.array(tensor.get_shape().as_list()))
            total_free_parameters += free_parameters
            self.print_and_log('{0: <27}'.format(str(tensor.name)) + ' ' + str(free_parameters))
        self.print_and_log('\n{0: <25}'.format('total free parameters:') + str(total_free_parameters))

class LeNet(Net):
    def __init__(self, args, print_and_log, save_figure):
        super().__init__(args, print_and_log, save_figure, tf.compat.v1.train.MomentumOptimizer, {'momentum': 0.9})

    def build_network(self):

        conv_1 = self.conv_layer('conv_1', self.x, 5, 6, padding='VALID')
        conv_1 = tf.nn.max_pool(conv_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='conv_1_pool')
        
        conv_2 = self.conv_layer('conv_2', conv_1, 5, 16, padding='VALID')
        conv_2 = tf.nn.max_pool(conv_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='conv_2_pool')
        flat   = tf.contrib.layers.flatten(conv_2)
        
        dense_1 = self.dense_layer('fc_1', flat,    120)
        dense_2 = self.dense_layer('fc_2', dense_1, 84)
        output  = self.dense_layer('fc_3', dense_2, self.args['num_classes'], False)

        return output
        
class ConvNet(Net):
    def __init__(self, args, print_and_log, save_figure):
        super().__init__(args, print_and_log, save_figure, tf.compat.v1.train.AdamOptimizer, {})

    def build_network(self):
        cur_image_size = self.args['input_shape'][0]        
     
        conv_1_1 = self.conv_layer('conv_1_1', self.x,   3, 96)
        conv_1_2 = self.conv_layer('conv_1_2', conv_1_1, 3, 96)       
        conv_1_3 = self.conv_layer('conv_1_3', conv_1_2, 3, 96) 
        conv_1_3 = tf.nn.max_pool2d(conv_1_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='conv_1_3_pool')
        cur_image_size /= 2
        
        conv_2_1 = self.conv_layer('conv_2_1', conv_1_3, 3, 192)
        conv_2_2 = self.conv_layer('conv_2_2', conv_2_1, 3, 192)
        conv_2_3 = self.conv_layer('conv_2_3', conv_2_2, 3, 192)
        conv_2_3 = tf.nn.max_pool2d(conv_2_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='conv_2_3_pool')        
        cur_image_size /= 2

        conv_3_1 = self.conv_layer('conv_3_1', conv_2_3, 3, 192)        
        conv_3_2 = self.conv_layer('conv_3_2', conv_3_1, 3, 192)
        conv_3_3 = self.conv_layer('conv_3_3', conv_3_2, 1, self.args['num_classes'])
                     
        global_averaging = tf.nn.avg_pool2d(
                 value   = conv_3_3,
                 ksize   = [1, cur_image_size, cur_image_size, 1],
                 strides = [1, 1, 1, 1],
                 padding = 'VALID')
        output = tf.squeeze(global_averaging, axis=(1,2))  
    
        return output
        
        
