# Hierarchical Co-occurrence Network with Prototype Loss for Few-shot Learning (PyTorch)

**Hierarchical Co-occurrence Network** from *Co-occurrence Feature Learning from Skeleton Data for Action Recognition and Detection with Hierarchical Aggregation*
[[Arxiv Preprint]](https://arxiv.org/abs/1804.06055)

**Prototype loss training procedure** from *Prototypical Networks for Few-shot Learning* https://arxiv.org/abs/1703.05175)

### Contributions
1) PyTorch reimplementation of **Hierarchical Co-occurrence Network (HCN)** 
2) Application of **Prototype loss** during training of HCN
3) Experiments showing that training with prototype loss can **improve the over all accuracy**


Software architecture inspired by:
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
