# Sparse Vessel Masks

This is a repository for the creation of sparse masks to train the segmentation of vessels.


# Example Data

We created an Example dataset using the Totalsegmentator Dataset.

1. Download the totalsegmentator dataset from here: https://zenodo.org/records/14710732
2. Unpack the dataset
3. set environment variable ```sparseVesselMasks_raw``` to the ```<repository_path>/example_dataset```
3. set environment variable ```PYTHONPATH```` to <<repository_path>>/src
4. run ```python <repository_path>/example_dataset/copy_relevant_image_data.py -i <path_to_total_segmentation_dataset>```
4. run ```python <repository_path>/src/sparselabel/entrypoints/create_sparse_label.py -d 1 -n 4 --no-wall --checkDataset``` (if you have sufficient RAM and cores you can set a higher -n to speed up the process)
5. You can see the created labels in ```<repository_path>/example_dataset/labelsTr``` and use them for network training.
6. If you want to train a nnUNet follow their documentation (https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md)
    - install nnUNetv2 (https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md)
    - set environment variable ```nnUNet_raw ``` to the ```<repository_path>/example_dataset```
    - set environment variable ```nnUNet_results ``` to the ```<repository_path>/nnUNet_results```
    - set environment variable ```nnUNet_preprocessed ``` to the ```<repository_path>/nnUNet_preprocessed```
    - run ```nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity```
    - run ```nnUNetv2_train DATASET_NAME_OR_ID 3d_fullres FOLD```