# PR-SDMLP

This code is for the paper title 'PR-SDMLP: Progressive Regularized Network via Shifted and Dynamic MLP for Jaw Cyst Segmentation in CBCT Images' 

submitted to IEEE Transactions on Emerging Topics in Computational Intelligence.

The corrsponding dataset can be downloaded from the website: https://pan.baidu.com/s/11EcV97RXeVnZx2mtngTUUA

#Note: You can download the datasets from the URLS before you send the Dataset End-User Agreement Instructions, we will send the password or the updating urls to you.

# How to use this code

1. Download nnUNet framework from the website:https://github.com/MIC-DKFZ/nnUNet, and use the following commands:

git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .

2. Download the jaw cyst dataset from https://pan.baidu.com/s/13KMnDlZWgeyzq0XVI10j6Q and password: ztzu.

3. The above files including PR_SDMLP.py and PR_SDMLP_Encoder.py are put into the fold: nnunetv2\training\nnUNetTrainer\variants\network_architecture.

4. The file 'nnUNetTrainer.py' is replaced, which is located in the folder 'nnunetv2\training\nnUNetTrainer\'.

5. put the file named as 'run_training.py' into the folder 'nnunetv2\run'

6. run the command 'python run_training.py'.

If you encounter any problems, please do not hesitate to contact us.

The corresponding pretrained model can be downloaded from the URL: https://drive.google.com/file/d/1jd-TKFczJ_WjENbqfOtGP7-3jFot13Et/view?usp=drive_link
