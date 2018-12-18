# Hierarchical Co-occurrence Network with Prototype Loss for Few-shot Learning (PyTorch)

**Hierarchical Co-occurrence Network** from *Co-occurrence Feature Learning from Skeleton Data for Action Recognition and Detection with Hierarchical Aggregation*
[[Arxiv Preprint]](https://arxiv.org/abs/1804.06055)

**Prototype loss training procedure** from *Prototypical Networks for Few-shot Learning* https://arxiv.org/abs/1703.05175)

Contributions:
1) PyTorch reimplementation of **Hierarchical Co-occurrence Network (HCN)** 
2) Application of **Prototype loss** during training of HCN
3) Experiments showing that training with prototype loss can **improve the over all accuracy**


software architecture inspired by:
   *  https://github.com/yysijie/st-gcn
   *  https://github.com/huguyuehuhu/HCN-pytorch
   *  https://github.com/jakesnell/prototypical-networks

## Data set
We used the [NTU RGB+D Action Recognition Dataset](http://rose1.ntu.edu.sg/datasets/actionrecognition.asp) [[Arxiv Preprint]](https://arxiv.org/abs/1604.02808) for our experiments.
The data set has to be downloaded and extracted under ```./data/NTU-RGB-D```.

## Prerequisites

The code is based on **Python 3.6**. All dependencies are listed in  [environment.yml](./environment.yml).
```commandline
conda env create -f environment.yml
```

## Usage
### Data preparation
##### NTU RGB+D
To transform raw NTU RGB+D data into numpy array (memmap format ) by this command:
```commandline
python ./tools/ntu_gendata.py --data_path ./data/NTU-RGB-D/nturgb+d_skeletons --out_folder 'data/NTU-RGB-D'  # default setting
python ./tools/ntu_gendata.py --data_path <path for raw skeleton dataset> --out_folder <path for new dataset>  # custom setting
```
##### Other Datasets
Not supported now.

### Training
Experiments can be configured via configuration files (```./config```) or via command line.

##### Train standard HCN
```commandline
$ python run_hcn.py -c config/HCN.yaml --use_gpu -w work_dir/HCN
```

##### Train HCN with prototype loss
```commandline
$ python run_protonet.py -c config/ProtoNet.yaml --use_gpu -w work_dir/Prototype
```

##### Command line help
```commandline
$ python run_hcn.py --help
usage: run_hcn.py [-h] [-w WORK_DIR] [-c CONFIG] [--use_gpu] [--debug]
                  [--phase PHASE] [--save_result] [--start_epoch START_EPOCH]
                  [--num_epoch NUM_EPOCH] [--device DEVICE [DEVICE ...]]
                  [--log_interval LOG_INTERVAL]
                  [--save_interval SAVE_INTERVAL]
                  [--eval_interval EVAL_INTERVAL] [--save_log] [--print_log]
                  [--show_topk SHOW_TOPK [SHOW_TOPK ...]] [--model MODEL]
                  [--model_args MODEL_ARGS] [--weights WEIGHTS]
                  [--ignore_weights IGNORE_WEIGHTS [IGNORE_WEIGHTS ...]]
                  [--loss LOSS] [--optimizer OPTIMIZER]
                  [--optimizer_args OPTIMIZER_ARGS] [--scheduler SCHEDULER]
                  [--scheduler_args SCHEDULER_ARGS] [--feeder FEEDER]
                  [--train_feeder_args TRAIN_FEEDER_ARGS]
                  [--test_feeder_args TEST_FEEDER_ARGS]
                  [--train_batch_size TRAIN_BATCH_SIZE]
                  [--test_batch_size TEST_BATCH_SIZE]
                  [--num_worker NUM_WORKER]

Processor

optional arguments:
  -h, --help            show this help message and exit
  -w WORK_DIR, --work_dir WORK_DIR
                        the work folder for storing results
  -c CONFIG, --config CONFIG
                        path to the configuration file
  --use_gpu             use GPUs or not
  --debug               less data, faster loading
  --phase PHASE         train or test
  --save_result         save output of model
  --start_epoch START_EPOCH
                        start training from which epoch
  --num_epoch NUM_EPOCH
                        stop training in which epoch
  --device DEVICE [DEVICE ...]
                        indexes of GPUs for training or testing
  --log_interval LOG_INTERVAL
                        interval for printing messages (#iteration)
  --save_interval SAVE_INTERVAL
                        interval for storing models (#iteration)
  --eval_interval EVAL_INTERVAL
                        interval for evaluating models (#iteration)
  --save_log            save logging or not
  --print_log           print logging or not
  --show_topk SHOW_TOPK [SHOW_TOPK ...]
                        show top-k accuracies
  --model MODEL         type of model
  --model_args MODEL_ARGS
                        arguments for model
  --weights WEIGHTS     weights for model initialization
  --ignore_weights IGNORE_WEIGHTS [IGNORE_WEIGHTS ...]
                        ignored weights during initialization
  --loss LOSS           type of loss function
  --optimizer OPTIMIZER
                        type of optimizer
  --optimizer_args OPTIMIZER_ARGS
                        arguments for optimizer
  --scheduler SCHEDULER
                        type of scheduler
  --scheduler_args SCHEDULER_ARGS
                        arguments for scheduler
  --feeder FEEDER       type of data loader
  --train_feeder_args TRAIN_FEEDER_ARGS
                        arguments for training data loader
  --test_feeder_args TEST_FEEDER_ARGS
                        arguments for test data loader
  --train_batch_size TRAIN_BATCH_SIZE
                        batch size for training
  --test_batch_size TEST_BATCH_SIZE
                        batch size for test
  --num_worker NUM_WORKER
                        number of workers per gpu for data loader
```


## Results
 Run tensorboard to view the results.
 ```commandline
 $ tensorboard --logdir ./work_dir
 ```
|                | Vanilla HCN | Prototype HCN |
| -------------- |  ---------- | ------------- |
| Accuracy       | 88.12 %     | **90.55 %**   |
| Top-2 accuracy | 94.91 %     | **96.98 %**   |
| Top-5 accuracy | 98.44 %     | **99.31 %**   |

 
![Screenshot TensorBoard][tensorboard]

![Confusion matrixes - Vanilla HCN vs Prototype HCN][confusion_matrixes]
 
### Citing Hierarchical Co-occurrence Network with Prototype Loss for Few-shot Learning
If you use Hierarchical Co-occurrence Network with Prototype Loss for Few-shot Learning in a scientific publication, I would appreciate references to the source code.

Biblatex entry:

```latex
@misc{Hierarchical Co-occurrence Network with Prototype Loss for Few-shot Learning,
  author = {Strobel, Max},
  title = {HCN-PrototypeLoss-PyTorch},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/maxstrobel/HCN-PrototypeLoss-PyTorch}}
}
```


[tensorboard]: img/tensorboard.png
[confusion_matrixes]: img/confusion_matrixes.png
