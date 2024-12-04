# PR-SDMLP

This code is for the paper title 'PR-SDMLP: Progressive Regularized Network via Shifted and Dynamic MLP for Jaw Cyst Segmentation in CBCT Images' 

submitted to IEEE Transactions on Emerging Topics in Computational Intelligence.

The corrsponding dataset can be downloaded from the website: https://pan.baidu.com/s/13KMnDlZWgeyzq0XVI10j6Q and password: ztzu.

# How to use this code

1. Download nnUNet framework from the website:https://github.com/MIC-DKFZ/nnUNet, and use the following commands:

git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .

2. Download the jaw cyst dataset from https://pan.baidu.com/s/13KMnDlZWgeyzq0XVI10j6Q and password: ztzu.

3. The above files except for nnUNetTrainer.py are put into the fold: nnunetv2\training\nnUNetTrainer\variants\network_architecture.

4. In the meanwhile, you replace the nnUNetTrainer class file with uploaded file named as 'nnUNetTrainer.py.
