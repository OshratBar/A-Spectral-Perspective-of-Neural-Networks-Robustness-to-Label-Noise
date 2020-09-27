# A-Spectral-Perspective-of-Neural-Networks-Robustness-to-Label-Noise 

Train network (AllCNN / LeNet) with noisy datasets, using different regularization methods. <br/>
Code creates the requiered dataset, trains the network for "repeat_num"  times, and saves run logs and results in a dedicated folder.<br/>

Data:<br/>
dataset:          cifar10, cifar100 or mnist                  (type=str,   default='cifar10')<br/>
noise_type:       uniform_noise or flip_noise                 (type=str,   default='uniform_noise')<br/>
noise_rate:       rate of corrupted samples                   (type=float, default=0.0)<br/>
validation_ratio: rate of validation data (out of train data) (type=float, default=0.1)<br/>
preprocess_mode:  range_0_to_1 or mean_substract              (type=str,   default='range_0_to_1')<br/>
   
Train:<br/>
do_sn:         whether or not to spectrally normalize the network weights<br/>
wd_coef:       coefficient of L2 loss term       (type=float, default=0.0)<br/>
jacob_coef:    coefficient of jacobian loss term (type=float, default=0.0)<br/>
entropy_coef:  coefficient of entropy loss term  (type=float, default=0.0)<br/>
epochs_num:    number of epochs in each train run(type=int,   default=30)<br/>
total_repeats: number of train runs              (type=int,   default=5)<br/>
GPU_num:       number of GPU to use              (type=int,   default=0)<br/>

Examples:<br/>
python CodeManager.py --noise_rate 0.4 --noise_type flip_noise    --dataset mnist    --wd_coef 1e-4 --do_sn --GPU_num 0<br/>
python CodeManager.py --noise_rate 0.3 --noise_type uniform_noise --dataset cifar10  --wd_coef 1e-4 --do_sn --entropy_coef 1 --epochs_num 20 --GPU_num 1<br/>
python CodeManager.py --noise_rate 0.5 --noise_type flip_noise    --dataset cifar100 --wd_coef 1e-4 --epochs_num 35 --GPU_num 2<br/>
python CodeManager.py --noise_rate 0.7 --noise_type uniform_noise --dataset cifar10  --wd_coef 1e-5 --jacob_coef 1e-3 --GPU_num 3<br/>


