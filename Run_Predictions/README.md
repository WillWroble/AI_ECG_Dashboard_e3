## License

See license tab for details.

## Model

The model receives an input tensor with dimension `(N, 2048, 12)`, and returns an output tensor with dimension `(N, 5)`, for which `N` is the batch size

- **input**: `shape = (N, 2048, 12)`. The input tensor should contain the  `2048` points of the ECG tracings
sampled at `250Hz` (i.e., a signal of approximately 8 seconds). The last dimension of the 
tensor contains points of the 12 different leads. The leads are ordered in the following order: 
`I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6`. 

- **output**: `shape = (N, 5)`. Each entry contains a probability between 0 and 1, and can be understood as the
probability of a given abnormality to be present. The abnormalities it predicts are in this order: LVEF ≤ 50, LVEF ≤ 45, LVEF ≤ 40, LVEF ≤ 35, LVEF ≤ 30. The abnormalities are not mutually exclusive, so the probabilities do not necessarily sum to one.

## Instructions:  
1. Obtain dependencies (from jm_environment.yml)
2. Run predictions (python predict.py ./NAME_OF_ECG_FILE.h5 ./LV_dysfunction_CHD.hdf5 --output_file collaborator_predictions.npy)
3. Run jupyter notebook (jupyter notebook CHOP_data.ipynb). Note there are a few placeholders with comments that you will have to revise with the names of your own files. Cell 1 corresponds to overall model performance, cell 2 corresponds to subgroup analysis. I recommend organizing your test data similar to the "fake_test_data.csv" file.
