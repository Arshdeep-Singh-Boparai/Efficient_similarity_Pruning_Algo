# Efficient_similarity_Pruning_Algo

This is a repo of our following paper,

Singh, Arshdeep, and Mark D. Plumbley. "Efficient Similarity-Based Passive Filter Pruning for Compressing CNNS." ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2023.

This is a repository containing scripts to prune CNNs using similarity-based pruning algorithm in a fast manner. 



# A brief overview of various Scripts

1) Efficient_Similarity_Pruning.py: To compute indexes of important filters for a given convolution layer in an efficient way by approximating the distance matrix or using complete distance matrix. The input to the script is the pre-trained weights. 

2) Plot_generation.py :  To generate Figures in the result and analysis section.

3) Fine_tuning_DCASE21_Net.py:  Given important filter indexes for all layers, Obtaining a pruned network and peforming fine-tuning.

# Folders

1) important_filter_indexes for VGGish_Net and DCASE21_Net: The important set of filters obtained using "Efficient_Similarity_Pruning.py", l1-norm based method and GM pruning method..


# Links

1. DCASE2018 dataset and VGGish_Net baseline (pre-trained weights, model with 64.69% accuracy):  https://drive.google.com/drive/folders/1b-eOYzNm2-IjTLf6jeaGr9twpd79LKFm?usp=sharing

2. DCASE2021 dataset and DCASE21_Net baseline (pre-trained weights, model with 48.58% accuracy): https://drive.google.com/drive/folders/1b-eOYzNm2-IjTLf6jeaGr9twpd79LKFm?usp=sharing

if above links does not work, please see below link

https://doi.org/10.5281/zenodo.7119930
