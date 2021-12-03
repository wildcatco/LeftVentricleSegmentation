## Installation
---
### environment
OS : ubuntu18.04  

### install with conda
conda create -n LVS python=3.8  
conda activate LVS  
pip install opencv-python  
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-lts  
pip install segmentation-models-pytorch albumentations  


## Train
---


## Test
---
Test one model
```
python test.py [checkpoint path]
```
Test ensemble of models
```
python test_ensemble.py [checkpoint1 checkopint2 ...]
```