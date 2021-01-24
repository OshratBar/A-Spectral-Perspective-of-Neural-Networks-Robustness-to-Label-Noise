# A-Spectral-Perspective-of-Neural-Networks-Robustness-to-Label-Noise 

Train network (AllCNN / LeNet) with noisy datasets, using different regularization methods. 
Code creates the requiered dataset, trains the network for "repeat_num"  times, and saves run logs and results in a dedicated folder.

**Data**

parameter        |description                                 |type  |default       
 --------------- | ------------------------------------------ | ---- | --------------
dataset          |cifar10, cifar100 or mnist                  |str   |'cifar10'      
noise_type       |uniform_noise or flip_noise                 |str   |'uniform_noise'
noise_rate       |rate of corrupted samples                   |float |0.0            
validation_ratio |rate of validation data (out of train data) |float |0.1             
preprocess_mode  |range_0_to_1 or mean_substract              |str   |'range_0_to_1' 


**Train**

parameter        |description                                 |type  |default       
 --------------- | ------------------------------------------ | ---- | --------------
do_sn         |whether or not to spectrally normalize the network weights |store_true||
wd_coef       |coefficient of L2 loss term                                |float |0.0
jacob_coef    |coefficient of jacobian loss term                          |float |0.0
entropy_coef  |coefficient of entropy loss term                           |float |0.0
epochs_num    |number of epochs in each train run                         |int   |30
total_repeats |number of train runs                                       |int   |5
GPU_num       |number of GPU to use                                       |int   |0

**Examples**
```console
python CodeManager.py --noise_rate 0.4 --noise_type flip_noise    --dataset mnist    --wd_coef 1e-4 --do_sn --GPU_num 0
python CodeManager.py --noise_rate 0.3 --noise_type uniform_noise --dataset cifar10  --wd_coef 1e-4 --do_sn --entropy_coef 1 --epochs_num 20 --GPU_num 1
python CodeManager.py --noise_rate 0.5 --noise_type flip_noise    --dataset cifar100 --wd_coef 1e-4 --epochs_num 35 --GPU_num 2
python CodeManager.py --noise_rate 0.7 --noise_type uniform_noise --dataset cifar10  --wd_coef 1e-5 --jacob_coef 1e-3 --GPU_num 3
```


