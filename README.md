# Efficient_similarity_Pruning_Algo

This is a repository containing scripts to prune CNNs using similarity-based pruning algorithm in a fast manner. 


# A brief overview of various Scripts

1) Efficient_Similarity_Pruning.py: To compute indexes of important filters for a given convolution layer in an efficient way by approximating the distance matrix or using complete distance matrix. The input to the script is the pre-trained weights. 

2) Plot_generation.py :  To generate Figures in the result and analysis section.

# Folders

1) important_filter_indexes: The important set of filters obtained using "Efficient_Similarity_Pruning.py", l1-norm based method and GM pruning method.
