"""
Merge EFD and keypoint pickles into the OpenGait sequence layout used by FreQait.

    EFD  (EFD-Analysis output) : <efd_root>/<ID>/<CAM>/<SEQ>.pkl         [T, 8, H, 4]
    pose                       : <pose_root>/<ID>/<CAM>/<SEQ>/*.pkl      [T, 17, 3]
                                 (or <pose_root>/<ID>/<CAM>/<SEQ>.pkl)
    output                     : <out_root>/<ID>/<CAM>/<SEQ>/00-efd.pkl
                                                            /01-pose.pkl

If both sequences have the same length they are symlinked, otherwise both are
trimmed to the shorter length and written as new files.

python datasets/merge_efd_pose.py --efd_root <...>/EFD8 --pose_root <...> --out_root <...>
"""
import argparse
import os
import pickle
from pathlib import Path

import numpy as np


def load(path):
    with open(path, 'rb') as f:
        return np.asarray(pickle.load(f))


def find_pose(pose_root, rel):
    seq_dir = pose_root / rel
    if seq_dir.is_dir():
        pkls = sorted(seq_dir.glob('*.pkl'))
        return pkls[0] if len(pkls) == 1 else None
    f = seq_dir.with_suffix('.pkl')
    return f if f.is_file() else None


def put(src, arr, dst, link):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if arr is None and link:
        os.symlink(src.resolve(), dst)
    else:
        with open(dst, 'wb') as f:
            pickle.dump(load(src) if arr is None else arr, f, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--efd_root', required=True, type=Path)
    ap.add_argument('--pose_root', required=True, type=Path)
    ap.add_argument('--out_root', required=True, type=Path)
    ap.add_argument('--copy', action='store_true', help='copy instead of symlink')
    ap.add_argument('--max_len_diff', type=int, default=5,
                    help='skip sequences whose EFD / pose lengths differ by more than this')
    args = ap.parse_args()

    n_ok = n_trim = n_skip = 0
    for efd_file in sorted(args.efd_root.rglob('*.pkl')):
        rel = efd_file.relative_to(args.efd_root).with_suffix('')          # ID/CAM/SEQ
        pose_file = find_pose(args.pose_root, rel)
        if pose_file is None:
            print('[skip] no (unique) pose pkl for', rel)
            n_skip += 1
            continue
        efd, pose = load(efd_file), load(pose_file)
        if efd.ndim != 4 or pose.ndim != 3 or pose.shape[1:] != (17, 3):
            print('[skip] unexpected shapes', rel, efd.shape, pose.shape)
            n_skip += 1
            continue
        T = min(len(efd), len(pose))
        if abs(len(efd) - len(pose)) > args.max_len_diff or T == 0:
            print('[skip] length mismatch', rel, len(efd), len(pose))
            n_skip += 1
            continue

        out_dir = args.out_root / rel
        out_dir.mkdir(parents=True, exist_ok=True)
        same = len(efd) == len(pose)
        put(efd_file, None if same else efd[:T], out_dir / '00-efd.pkl', not args.copy)
        put(pose_file, None if same else pose[:T], out_dir / '01-pose.pkl', not args.copy)
        n_ok += same
        n_trim += not same
    print('done: {} merged, {} trimmed, {} skipped -> {}'.format(n_ok, n_trim, n_skip, args.out_root))


if __name__ == '__main__':
    main()
