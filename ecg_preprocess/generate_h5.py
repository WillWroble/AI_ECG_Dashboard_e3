from ecg_preprocess import preprocess
from ecg_preprocess import read_ecg
import argparse
import h5py
import pandas as pd
import tqdm
import os
import sys


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description='Generate H5 from ECG records.')
    parser.add_argument('input_file', type=str, help='path to RECORDS file.')
    parser.add_argument('out_file', type=str, help='output H5 file.')
    parser.add_argument('--root_dir', type=str, help='Root dir for relative paths.')
    parser = preprocess.arg_parse_option(parser)
    parser = read_ecg.arg_parse_option(parser)
    args = parser.parse_args(argv)
    print(args)

    # Load RECORDS
    files = pd.read_csv(args.input_file, header=None).values.flatten().tolist()
    folder = args.root_dir if args.root_dir is not None else os.path.dirname(args.input_file)

    # Ensure parent of out_file exists
    out_parent = os.path.dirname(args.out_file) or "."
    os.makedirs(out_parent, exist_ok=True)

    # First pass: probe files to compute final count & shape
    good = []
    n_samples = n_leads = None
    for f in files:
        path = os.path.join(folder, f)
        try:
            ecg, sample_rate, leads = read_ecg.read_ecg(path, format=args.fmt)
            ecg_pre, new_rate, new_leads = preprocess.preprocess_ecg(
                ecg, sample_rate, leads,
                new_freq=args.new_freq, new_len=args.new_len,
                scale=args.scale, powerline=args.powerline,
                use_all_leads=args.use_all_leads, remove_baseline=args.remove_baseline
            )
            if n_samples is None:
                n_leads, n_samples = ecg_pre.shape
            good.append((path, ecg_pre))
        except Exception:
            print(f"[WARN] Skipping unreadable ECG: {path}")
            traceback.print_exc(limit=1)

    if not good:
        raise RuntimeError("No valid ECGs found; nothing to write to H5.")

    # Create H5 with final size
    with h5py.File(args.out_file, 'w') as h5f:
        x = h5f.create_dataset('tracings', (len(good), n_samples, n_leads), dtype='f8')
        for i, (_, ecg_pre) in enumerate(tqdm.tqdm(good)):
            x[i, :, :] = ecg_pre.T



if __name__ == '__main__':
    main()