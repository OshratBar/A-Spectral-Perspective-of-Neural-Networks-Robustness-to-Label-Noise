import glob
import os
import pickle
import argparse
import argunparse
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from Net import ConvNet, LeNet
import tensorflow as tf
        
class CodeManager():
    
    def __init__(self, parameters):
        self.P = parameters
        self.initialize_log_file()              
        self.print_and_log('\n** Parameters **\n')
        self.print_and_log_dict(parameters)
        
    def initialize_log_file(self):       
        log_files = glob.glob(self.P['logs_dir'] + '*.log')
        log_files_numbers = [int(file.split('/')[-1].split('.')[0]) for file in log_files] # server
        next_number = 1
        if len(log_files_numbers):
            next_number = max(log_files_numbers) + 1
        log_file_name = self.P['logs_dir'] + str(next_number) + '.log'                            
        self.log_file = open(log_file_name, 'a')
        print("\nopening log file", log_file_name)        
        self.print_and_log('repeat %d/%d'%(self.P['repeat_num'], self.P['total_repeats']))            
        self.P['train']['save_to_path'] = self.P['checkpoints_dir'] + 'run_' + str(next_number)        
        self.current_run_number = next_number       
       

    def go(self):    
        self.D   = pickle.load(open(self.P['data_file'], 'rb'))
        net      = LeNet if self.P['data_set']=='mnist' else ConvNet
        self.net = net(self.P['network'], self.print_and_log, self.save_figure)       
        train_dict = {
            'train_data'        : self.D['images']['train'],           
            'train_target'      : to_categorical(self.D['train_noisy_labels']),
            'train_true_target' : to_categorical(self.D['labels']['train']),
            'is_corrupt'        : self.D['train_is_corrupt'], 
            'valid_data'        : self.D['images']['valid'],
            'valid_target'      : to_categorical(self.D['labels']['valid']),
            'test_data'         : self.D['images']['test'],
            'test_target'       : to_categorical(self.D['labels']['test'])}
        train_dict.update(self.P['train']) 
        history   = self.net.train(train_dict)                
        eval_vals = self.eval_pred()  
        
        # Test 1 - train jacobian  
        jacob_batch = 500
        jacob_loss = np.empty(0)        
        for i in range(int(len(self.D['images']['train'])/jacob_batch)):
            jacob = self.net.session.run(self.net.jacob_loss, feed_dict={self.net.x:train_dict['train_data'][i*jacob_batch:(i+1)*jacob_batch]})             
            jacob_loss = np.concatenate((jacob_loss, jacob))
        eval_vals.append(np.mean(jacob_loss))
        
        # Test 2 - test jacobian
        jacob_batch = 500
        jacob_loss = np.empty(0)
        for i in range(int(len(self.D['images']['test'])/jacob_batch)):
            jacob = self.net.session.run(self.net.jacob_loss, feed_dict={self.net.x:self.D['images']['test'][i*jacob_batch:(i+1)*jacob_batch]})
            jacob_loss = np.concatenate((jacob_loss, jacob))
        eval_vals.append(np.mean(jacob_loss))
           
        # Test 3 - entropy 
        logits = self.net.predict(train_dict['train_data'], batch_size=1000)
        probs  = np.exp(logits)/np.sum(np.exp(logits), axis=1, keepdims=True)
        entropy = -np.sum(probs*np.log(probs), axis=-1)                     
        eval_vals.append(np.mean(entropy))         
      
        self.log_file.close()
        
        # save only latest checkpoint
        checkpoints     = glob.glob(self.P['checkpoints_dir'] + 'run_' + str(self.current_run_number) + '*')
        best_checkpoint = tf.train.latest_checkpoint(manager.P['checkpoints_dir'])
        for c in checkpoints:
            if best_checkpoint not in c:
                os.remove(c) 
             
        return np.array(eval_vals)[None, :], history
        
    def eval_pred(self):
        self.net.saver.restore(self.net.session, tf.train.latest_checkpoint(self.P['checkpoints_dir']))
        _, train_noisy_accuracy = self.net.evaluate_prediction(self.D['images']['train'], to_categorical(self.D['train_noisy_labels']), 1000) 
        _, train_accuracy       = self.net.evaluate_prediction(self.D['images']['train'], to_categorical(self.D['labels']['train']),    1000) 
        _, valid_accuracy       = self.net.evaluate_prediction(self.D['images']['valid'], to_categorical(self.D['labels']['valid']),    1000) 
        _, test_accuracy        = self.net.evaluate_prediction(self.D['images']['test'],  to_categorical(self.D['labels']['test']),     1000) 
        self.print_and_log('\nTrain noisy accuracy: ' + str(train_noisy_accuracy))
        self.print_and_log('Train accuracy: '         + str(train_accuracy))
        self.print_and_log('Valid accuracy: '         + str(valid_accuracy))
        self.print_and_log('Test accuracy:  '         + str(test_accuracy))
        return [train_noisy_accuracy, train_accuracy, valid_accuracy, test_accuracy]
        
 
    def print_and_log(self, message):       
        print(message)
        self.log_file.write(message + '\n')
        self.log_file.flush()
    
    def print_and_log_dict(self, dictionary, space=0):
        for key in dictionary.keys():
            if isinstance(dictionary[key], dict): 
                self.print_and_log(key+':')
                self.print_and_log_dict(dictionary[key], space+2)
            else:
                self.print_and_log(space*' ' + '{0: <20}'.format(key) + ': ' + str(dictionary[key]))  
    
    def save_figure(self, name):
        fig_name = self.P['figures_dir'] + name + str(self.current_run_number)
        plt.savefig(fig_name, dpi=100)
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Build noisy training set, train and evaluate')    
    parser.add_argument('--validation_ratio', type=float, default=0.1,             help='rate of validation data (out of train data)') 
    parser.add_argument('--noise_rate',       type=float, default=0.0,             help='rate of corrupted samples') 
    parser.add_argument('--noise_type',       type=str,   default='uniform_noise', help='uniform_noise or flip_noise')    
    parser.add_argument('--preprocess_mode',  type=str,   default='range_0_to_1',  help='range_0_to_1 or mean_substract') 
    parser.add_argument('--dataset',          type=str,   default='cifar10',       help='cifar10, cifar100 or mnist')
    parser.add_argument('--wd_coef',          type=float, default=0.0,             help='coefficient of L2 loss term')
    parser.add_argument('--jacob_coef',       type=float, default=0.0,             help='coefficient of jacobian loss term')
    parser.add_argument('--entropy_coef',     type=float, default=0.0,             help='coefficient of entropy loss term')
    parser.add_argument('--do_sn',            action='store_true',                 help='whether or not to spectrally normalize the network weights')
    parser.add_argument('--total_repeats',    type=int,   default=5,               help='number of train runs')
    parser.add_argument('--epochs_num',       type=int,   default=30,              help='number of epochs in each train run')
    parser.add_argument('--GPU_num',          type=int,   default=0,               help='number of GPU to use')

    args = parser.parse_args()
    print(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.GPU_num)   

    args_dict      = vars(args)
    data_args_dict = dict((key, args_dict[key]) for key in ('validation_ratio', 'noise_rate', 'noise_type', 'preprocess_mode', 'dataset'))
    print('\n -- Data --')
    os.system('python ./DataHandler.py ' + str(argunparse.ArgumentUnparser().unparse(**data_args_dict)))
    print('\n')

    regular_methods = []
    if args.wd_coef      : regular_methods.append('wd')
    if args.jacob_coef   : regular_methods.append('jacob')
    if args.entropy_coef : regular_methods.append('entropy')
    if args.do_sn        : regular_methods.append('sn')
    regular_method = '_'.join(regular_methods) if len(regular_methods) else 'no_reg'

    outputs_dir = '/data/oshrat/final/' + args.dataset + '/' + args.noise_type +'/%s/'%(args.noise_rate) + regular_method + '/'
    
    print(outputs_dir)

    parameters = {
    'outputs_dir'     : outputs_dir, 
    'logs_dir'        : outputs_dir + 'logs/',
    'checkpoints_dir' : outputs_dir + 'checkpoints/',
    'figures_dir'     : outputs_dir + 'figures/',
    'noise_type'      : args.noise_type,
    'regular_method'  : regular_method,
    'noise_rate'      : args.noise_rate,
    'total_repeats'   : args.total_repeats,
    'repeat_num'      : 0,
    'data_file'       : './data/%s-%.1f_valid-%.1f_%s-%s'%(args.dataset, args.validation_ratio, args.noise_rate, args.noise_type, args.preprocess_mode),
    'data_set'        : args.dataset,

    'network' : {
             'do_spectral_norm'  : args.do_sn,
             'entropy_coef'      : args.entropy_coef,
             'jacob_coef'        : args.jacob_coef,
             'weight_decay_coef' : args.wd_coef,
             'num_classes'       : 100 if args.dataset=='cifar100' else 10,
             'input_shape'       : (28, 28, 1) if args.dataset=='mnist' else (32, 32, 3)},
    'train' : {
             'train_batch_size'      : 32,
             'valid_batch_size'      : 1000,
             'save_to_path'          : '',
             'restore_from_path'     : '',
             'restore_initalization' : False,
             'epochs_num'            : args.epochs_num,
             'initial_lr'            : 0.01 if args.dataset=='mnist' else 0.001,
             'decrease_lr_factor'    : 0.1,
             'decrease_lr_every'     : 15 if args.dataset=='mnist' else 10}
        }
    
    if not os.path.isdir(parameters['outputs_dir']):
        os.makedirs(parameters['outputs_dir'])
        os.makedirs(parameters['logs_dir'] )       
        os.makedirs(parameters['checkpoints_dir'])            
        os.makedirs(parameters['figures_dir']) 
     
    scores = np.empty((0, 7))
    train_history = []
    for num in range(parameters['total_repeats']):
        parameters['repeat_num'] = num + 1
        manager = CodeManager(parameters)
        eval_vals, history = manager.go()
        train_history.append(history)
        scores = np.concatenate((scores, eval_vals), axis=0)
        print(scores)    
    print(np.mean(scores, axis=0))
    print(np.std (scores, axis=0))
    
    file = open(parameters['outputs_dir'] + 'scores', 'wb')
    pickle.dump(scores , file)
    file.close() 
    
    file = open(parameters['outputs_dir'] + 'train_history', 'wb')
    pickle.dump(train_history , file)
    file.close() 
    
    file = open(parameters['outputs_dir'] + 'scores.txt', 'w')
    file.write('train_noisy_accuracy  |  train_accuracy  |  valid_accuracy  |  test_accuracy  |  train_jacob |  test_jacob | entropy\n\n')
    for i in range(len(scores)):
        for s in scores[i]:
            file.write('%f  | '%(s))
        file.write('\n')
    file.write('\n\nmean\n')
    for s in (np.mean(scores, axis=0)):
        file.write('%f  |  '%(s))
    file.write('\n\nstd\n')
    for s in (np.std(scores, axis=0)):
        file.write('%f  |  '%(s))
    file.close()
        



