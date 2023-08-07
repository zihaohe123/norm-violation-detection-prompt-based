# CPL-NoViD: Context-Aware Prompt-based Learning for Norm Violation Detection in Online Communities

In this [paper](https://arxiv.org/abs/2305.09846) we try to detect norm violation on Reddit using context-aware prompt-based learning. The code is based on [OpenPrompt](https://github.com/thunlp/OpenPrompt) for prompt-based learning. I recommend getting yourself familiarized with this framework if you wanna fully understand how the code works.


## Usage
### Installation
```angular2html
pip install -r requirements
```
There are not a whole bunch of dependencies so the environment should be fairly easy to set up.

### Hardware requirement
It's recommended that you train the model on a Tesla V100 or better GPU, which takes ~9 hours for the complete training. If you have a less powerful GPU, reduce the batch size accordingly.


### Overall training
```angular2html
bash run_overall.sh
```

### Cross rule-type training
```angular2html
bash run_cross_rule_typ.sh
```


### Cross community training
```angular2html
bash run_cross_community.sh
```

### Few-shot training
```angular2html
bash run_few_shot.sh
```

## Citation
If you find this repository useful, please cite our paper.

```bibtex
@article{he2023cpl,
  title={CPL-NoViD: Context-Aware Prompt-based Learning for Norm Violation Detection in Online Communities},
  author={He, Zihao and May, Jonathan and Lerman, Kristina},
  journal={arXiv preprint arXiv:2305.09846},
  year={2023}
}
```