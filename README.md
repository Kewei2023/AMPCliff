# AMPCliff

testing plenty of models for AMP activity cliff prediction

## Getting Started

### Dependencies

```
conda env create -f environment.yml
```

### Runing

#### Deep Learning Method
```
sbatch --gpus=1 downstream_train_.sh
```
#### Machine Learning Method
```
sbatch --gpus=1 machine_learning_train.sh
```

