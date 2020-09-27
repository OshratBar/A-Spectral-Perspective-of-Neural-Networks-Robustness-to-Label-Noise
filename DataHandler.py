from keras.datasets import cifar10, cifar100, mnist
import numpy as np
import argparse
import pickle
      

def split_train_validation(images_dict, labels_dict, validation_ratio):           
    images = images_dict['train']
    labels = labels_dict['train']   
    is_validation = np.zeros((len(labels)), np.bool)    
    for class_num in np.unique(labels):
        class_idx  = np.where(labels == class_num)[0]
        class_size = len(class_idx)
        valid_size = int(validation_ratio * class_size)
        is_validation[class_idx[:valid_size]] = True
    images_dict['valid'] = images[is_validation,...]
    labels_dict['valid'] = labels[is_validation,...]
    images_dict['train'] = images[~is_validation,...]
    labels_dict['train'] = labels[~is_validation,...]   
    return images_dict, labels_dict   
            
def corrupt_labels(noise_type, noise_rate, labels, dataset):                 
    sample_size  = len(labels) 
    is_corrupt   = np.zeros((sample_size), dtype=np.bool)
    noisy_labels = labels.copy()          
    if noise_type == 'uniform_noise':            
         print('corrupt labels by uniform noise, noise rate: ' + str(noise_rate))     
         num_classes  = len(np.unique(labels))                                             
         corrupt_size = int(sample_size * noise_rate)             
         corrupt_idx  = np.random.choice(np.arange(sample_size), corrupt_size, replace=False)
         is_corrupt[corrupt_idx]   = True                                         
         noisy_labels[corrupt_idx] = np.random.choice(num_classes, corrupt_size)                                
    elif noise_type == 'flip_noise':
        print('corrupt labels by flip noise, noise rate: ' + str(noise_rate))
        if dataset == 'cifar10':
            orig_class = [2, 3, 4, 5, 9] # [bird,     cat, deer,  dog, truck]
            dest_class = [0, 5, 7, 3, 1] # [airplane, dog, horse, cat, automobile]
        elif dataset == 'cifar100':
            orig_class = np.arange(100) 
            dest_class = orig_class + 1 
            idx = np.arange(4, 100, 5)
            dest_class[idx] = dest_class[idx] - 5
        elif dataset == 'mnist':
            orig_class = [1, 2, 3, 4, 5, 6, 7, 9] 
            dest_class = [7, 3, 2, 9, 6, 5, 1, 4] 
        for orig, dest in zip(orig_class, dest_class):
            class_idx = np.where(labels == orig)[0]
            np.random.shuffle(class_idx)
            corrupt_class_idx = class_idx[:int(noise_rate*len(class_idx))]
            is_corrupt[corrupt_class_idx] = True              
            noisy_labels[corrupt_class_idx] = dest
    return noisy_labels, is_corrupt 
   
            
def preprocess_images(images, mode):
    if mode == 'mean_substract':      
        mean = np.mean(images['train'], axis=0, keepdims=True)           
        for role in images.keys():           
            images[role] = (images[role] - mean).astype(np.float32)
    elif mode == 'range_0_to_1':
        brightest = np.iinfo(images['train'].dtype).max # integer info: maximum possible value  
        for role in images.keys():           
            images[role] = (images[role] / brightest).astype(np.float32)
    return images
        

def main():    
    np.random.seed(0)
    parser = argparse.ArgumentParser(description='Build noisy training set')    
    parser.add_argument('--validation_ratio', type=float, default=0.1,             help='rate of validation data (out of train data)') 
    parser.add_argument('--noise_rate',       type=float, default=0.0,             help='rate of corrupted samples') 
    parser.add_argument('--noise_type',       type=str,   default='uniform_noise', help='uniform_noise or flip_noise')    
    parser.add_argument('--preprocess_mode',  type=str,   default='range_0_to_1',  help='range_0_to_1 or mean_substract') 
    parser.add_argument('--dataset',          type=str,   default='cifar10',       help='cifar10, cifar100 or mnist')
    args = parser.parse_args()
    print(args)      
        
    # load data
    data   = {}
    images = {}
    labels = {}
    if args.dataset == 'cifar10':
        (images['train'], labels['train']), (images['test'], labels['test']) = cifar10.load_data()
    elif args.dataset == 'cifar100':
        (images['train'], labels['train']), (images['test'], labels['test']) = cifar100.load_data()
    elif args.dataset == 'mnist':
        (images['train'], labels['train']), (images['test'], labels['test']) = mnist.load_data()
        images['train'] = images['train'][..., None]
        images['test']  = images['test'] [..., None]
    print(images['test'].shape)
    labels['train'] = np.squeeze(labels['train'])
    labels['test']  = np.squeeze(labels['test'])
    
    # shuffle train data 
    idx = np.arange(len(images['train']))
    np.random.shuffle(idx)
    images['train'] = images['train'][idx]
    labels['train'] = labels['train'][idx]
    
    # seperate validation data out of train data
    images, data['labels'] = split_train_validation(images, labels, args.validation_ratio)       
    print('total train samples:      %d' % images['train'].shape[0])
    print('total validation samples: %d' % images['valid'].shape[0])
    print('total test samples:       %d' % images['test'].shape[0])
        
    # normalize images according to training data
    data['images'] = preprocess_images(images, args.preprocess_mode)
    
    # corrupt train labels
    data['train_noisy_labels'], data['train_is_corrupt'] = corrupt_labels(args.noise_type, args.noise_rate, data['labels']['train'], args.dataset)

    file = open('./data/%s-%.1f_valid-%.1f_%s-%s'%(args.dataset, args.validation_ratio, args.noise_rate, args.noise_type, args.preprocess_mode), 'wb')
    pickle.dump(data, file)  
    file.close()
     
if __name__ == '__main__':
    main()
   
        
 
    
    
    


