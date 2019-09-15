# Meta-Weight-Net
NeurIPS'19: Meta-Weight-Net: Learning an Explicit Mapping For Sample Weighting (Official Pytorch implementation for class-imbalance).
The implementation of noisy labels is available at https://github.com/xjtushujun/Meta-weight-net.


================================================================================================================================================================


This is the code for the paper:
[Meta-Weight-Net: Learning an Explicit Mapping For Sample Weighting](https://arxiv.org/abs/1902.07379)  
Jun Shu, Qi Xie, Lixuan Yi, Qian Zhao, Sanping Zhou, Zongben Xu, Deyu Meng*
To be presented at [NeurIPS 2019](https://nips.cc/Conferences/2019/).  

If you find this code useful in your research then please cite  
```bash
@inproceedings{han2018coteaching,
  title={Meta-Weight-Net: Learning an Explicit Mapping For Sample Weighting},
  author={Shu, Jun and Xie, Qi and Yi, Lixuan and Zhao, Qian and Zhou, Sanping and Xu, Zongben and Meng, Deyu},
  booktitle={NeurIPS},
  year={2019}
}
``` 


## Setups
The requiring environment is as bellow:  

- Linux 
- Python 3+
- PyTorch 0.4.0 
- Torchvision 0.2.0


## Running Meta-Weight-Net on benchmark datasets (CIFAR-10 and CIFAR-100).
Here is an example:
```bash
python meta-weight-net-class-imbalance.py --dataset cifar100 --num_classes 100 --imb_factor 0.01
```



## Acknowledgements
We thank the Pytorch implementation on class-balanced-loss(https://github.com/richardaecn/class-balanced-loss) and learning-to-reweight-examples(https://github.com/danieltan07/learning-to-reweight-examples).


Contact: Jun Shu (xjtushujun@gmail.com); Deyu Meng(dymeng@mail.xjtu.edu.cn).




