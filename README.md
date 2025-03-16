# [ICASSP'25] HGSC
Official Pytorch implementation of **A Hierarchical Compression Technique for 3D Gaussian Splatting Compression**.
[[`Arxiv`]([https://arxiv.org/pdf/2403.14530](https://arxiv.org/abs/2411.06976))]  [[`Github`]([https://github.com/YihangChen-ee/HAC](https://github.com/H-Huang774/HGSC))]
## Installation
```
conda create -n hgsc python=3.8
conda activate hgsc
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

# Clone this repository
git clone https://github.com/H-Huang774/HGSC.git
cd HGSC

# Install submodules
pip install dgmesh/submodules/diff-gaussian-rasterization
pip install dgmesh/submodules/simple-knn

# Install other dependencies
pip install -r requirements.txt
```
