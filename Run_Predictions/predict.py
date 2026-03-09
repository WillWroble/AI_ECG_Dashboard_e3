import numpy as np
import warnings
import argparse
import sys
import os
warnings.filterwarnings("ignore")
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from Run_Predictions.datasets import ECGSequence


def predict(input_file, model_file, output_file, dataset_name='tracings', bs=32):
    """
    Loads a model and runs predictions on an HDF5 file,
    ensuring the file handle is properly closed.
    """
    print(f"Loading model from {model_file} and data from {input_file}...")
    
    # Load the Keras model first
    model = load_model(model_file, compile=False)
    model.compile(loss='binary_crossentropy', optimizer=Adam())

    # Use a 'with' statement to create and manage the ECGSequence object.
    # This ensures its __del__ method (and thus the file.close() call) is
    # triggered as soon as the 'with' block is exited.
    with ECGSequence(input_file, dataset_name, batch_size=bs) as seq:
        y_score = model.predict(seq, verbose=1)

    # Save the predictions to a .npy file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.save(output_file, y_score)
    print("Output predictions saved to", output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get performance on test set from hdf5')
    parser.add_argument('path_to_hdf5', type=str,
                        help='path to hdf5 file containing tracings')
    parser.add_argument('path_to_model',
                        help='file containing training model.')
    parser.add_argument('--dataset_name', type=str, default='tracings',
                        help='name of the hdf5 dataset containing tracings')
    parser.add_argument('--output_file', default="./dnn_output.npy",
                        help='output csv file.')
    parser.add_argument('--bs', type=int, default=32,
                        help='Batch size.')

    args, unk = parser.parse_known_args()
    if unk:
        warnings.warn("Unknown arguments:" + str(unk) + ".")

    # Call the main function
    predict(
        input_file=args.path_to_hdf5,
        model_file=args.path_to_model,
        output_file=args.output_file,
        dataset_name=args.dataset_name,
        bs=args.bs
    )