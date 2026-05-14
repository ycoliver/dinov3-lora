#! /usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import sys
import time

import cv2
import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent
SRC_ROOT = CURRENT_DIR.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DATA_ROOT = SRC_ROOT / 'datasets'
DEFAULT_SCANNET_PAIRS = DATA_ROOT / 'scannet_with_gt.txt'
OUTPUT_ROOT = SRC_ROOT / 'evaluate'

def ensure_third_party_superglue_on_path():
    superglue_path = SRC_ROOT / 'Superglue'
    if str(superglue_path) not in sys.path:
        sys.path.insert(0, str(superglue_path))

DEFAULT_OUTPUT_DIR = OUTPUT_ROOT / 'Superglue_origin_loransac' / 'Superglue_csv_usac_eval'

def load_superglue_utils():
    ensure_third_party_superglue_on_path()

    from models.utils import (  # noqa: E402
        compute_epipolar_error,
        compute_pose_error,
        pose_auc,
        read_image,
        rotate_intrinsics,
        rotate_pose_inplane,
        scale_intrinsics,
    )

    return (
        compute_epipolar_error,
        compute_pose_error,
        pose_auc,
        read_image,
        rotate_intrinsics,
        rotate_pose_inplane,
        scale_intrinsics,
    )


def resolve_usac_method(method_name):
    method = getattr(cv2, method_name, None)
    if method is None:
        raise AttributeError('cv2 does not provide method {}'.format(method_name))
    return method


def find_essential_mat_usac(kpts0, kpts1, norm_thresh, conf, max_iters, method_name):
    camera_matrix = np.eye(3, dtype=np.float64)
    method = resolve_usac_method(method_name)
    kwargs = {
        'threshold': norm_thresh,
        'prob': conf,
        'method': method,
    }

    if max_iters is not None and max_iters > 0:
        try:
            return cv2.findEssentialMat(
                kpts0,
                kpts1,
                camera_matrix,
                maxIters=max_iters,
                **kwargs,
            )
        except TypeError:
            pass

        try:
            return cv2.findEssentialMat(
                kpts0,
                kpts1,
                camera_matrix,
                method,
                conf,
                norm_thresh,
                max_iters,
            )
        except TypeError:
            pass

    return cv2.findEssentialMat(kpts0, kpts1, camera_matrix, **kwargs)


def estimate_pose_usac(
        kpts0,
        kpts1,
        K0,
        K1,
        thresh,
        conf=0.99999,
        max_iters=None,
        method_name='USAC_DEFAULT',
        return_pose_data=False):
    if len(kpts0) < 5:
        return None

    f_mean = np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])
    norm_thresh = thresh / f_mean

    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    E, mask = find_essential_mat_usac(
        kpts0,
        kpts1,
        norm_thresh,
        conf,
        max_iters,
        method_name,
    )
    if E is None or mask is None:
        return None

    best_num_inliers = 0
    ret = None
    num_essential_mats = E.shape[0] // 3
    for single_E in np.split(E, num_essential_mats):
        n, R, t, _ = cv2.recoverPose(
            single_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            best_num_inliers = n
            if return_pose_data:
                ret = {
                    'R': R,
                    't': t[:, 0],
                    'inliers': mask.ravel() > 0,
                    'E': single_E,
                    'num_inliers': int(n),
                }
            else:
                ret = (R, t[:, 0], mask.ravel() > 0)
    return ret


class ProgressBar:
    def __init__(self, total, width=32):
        self.total = max(int(total), 0)
        self.width = max(int(width), 10)
        self.current = 0
        self.start_time = time.time()
        self.last_length = 0

    def update(self, current, extra=''):
        self.current = min(max(int(current), 0), self.total)
        if self.total == 0:
            percent = 100.0
            filled = self.width
        else:
            percent = 100.0 * self.current / self.total
            filled = int(self.width * self.current / self.total)
        bar = '#' * filled + '-' * (self.width - filled)
        elapsed = time.time() - self.start_time
        message = '\r[{}] {}/{} {:5.1f}% elapsed {:6.1f}s'.format(
            bar, self.current, self.total, percent, elapsed)
        if extra:
            if len(extra) > 48:
                extra = extra[:45] + '...'
            message += ' {}'.format(extra)
        padded_message = message
        if len(message) < self.last_length:
            padded_message += ' ' * (self.last_length - len(message))
        self.last_length = len(message)
        sys.stdout.write(padded_message)
        sys.stdout.flush()

    def close(self):
        if self.last_length > 0:
            sys.stdout.write('\n')
        sys.stdout.flush()


def image_output_id(name):
    path = Path(name)
    scene = next((part for part in path.parts if part.startswith('scene')), None)
    if scene is not None:
        return '{}_{}'.format(scene, path.stem)

    parent_parts = [part for part in path.parts[:-1] if part not in ('', '.')]
    if parent_parts:
        return '{}_{}'.format('_'.join(parent_parts), path.stem)

    return path.stem


def pair_output_id(name0, name1):
    return '{}_{}'.format(image_output_id(name0), image_output_id(name1))


def format_evaluation_summary(num_pairs, aucs, precision):
    return '\n'.join([
        'Evaluation Results (mean over {} pairs):'.format(num_pairs),
        'AUC@5\t AUC@10\t AUC@20\t Prec\t',
        '{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t'.format(
            aucs[0], aucs[1], aucs[2], precision),
        '',
    ])


def save_evaluation_summary(output_dir, num_pairs, aucs, precision):
    summary_text = format_evaluation_summary(num_pairs, aucs, precision)
    summary_path = output_dir / 'evaluation_results.txt'
    summary_path.write_text(summary_text)

    summary_json = {
        'num_pairs': num_pairs,
        'auc@5': aucs[0],
        'auc@10': aucs[1],
        'auc@20': aucs[2],
        'precision': precision,
    }
    summary_json_path = output_dir / 'evaluation_results.json'
    summary_json_path.write_text(json.dumps(summary_json, indent=2))

    return summary_text


def normalize_csv_stem(stem):
    for suffix in ('_matches', '_sinkhorn_scores', '_score_matrix'):
        if stem.endswith(suffix):
            return stem[:-len(suffix)]
    return stem


def build_csv_index(input_csv_dir):
    csv_index = {}
    for csv_path in sorted(input_csv_dir.rglob('*.csv')):
        csv_index.setdefault(normalize_csv_stem(csv_path.stem), []).append(csv_path)
    return csv_index


def resolve_csv_path(input_csv_dir, csv_index, pair_id):
    candidates = [
        input_csv_dir / '{}.csv'.format(pair_id),
        input_csv_dir / '{}_matches.csv'.format(pair_id),
        input_csv_dir / '{}_sinkhorn_scores.csv'.format(pair_id),
        input_csv_dir / '{}_score_matrix.csv'.format(pair_id),
    ]
    for candidate in candidates:
        import os, sys
        resolved = str(candidate.resolve())
        # Windows long-path prefix; harmless to skip on POSIX.
        safe_path = ("\\\\?\\" + resolved) if sys.platform.startswith("win") else resolved
        if os.path.exists(safe_path):
            return candidate

    indexed = csv_index.get(pair_id, [])
    if len(indexed) == 1:
        return indexed[0]
    if len(indexed) > 1:
        raise ValueError(
            'Ambiguous CSV for pair_id {}: {}'.format(
                pair_id, [str(path) for path in indexed]))

    raise FileNotFoundError(
        'Cannot find CSV for pair_id {} under {}. Tried: {}'.format(
            pair_id, input_csv_dir, [path.name for path in candidates]))


def load_csv_matches(csv_path):
    required_fields = {'left_idx', 'right_idx', 'score', 'x1', 'y1', 'x2', 'y2'}
    mkpts0 = []
    mkpts1 = []
    scores = []
    total_rows = 0

    import sys as _sys
    _resolved = str(csv_path.resolve())
    _open_path = ("\\\\?\\" + _resolved) if _sys.platform.startswith("win") else _resolved
    with open(_open_path, 'r', newline='') as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError('Missing CSV header in {}'.format(csv_path))
        missing = required_fields.difference(fieldnames)
        if missing:
            raise ValueError('Missing CSV fields {} in {}'.format(sorted(missing), csv_path))

        for row in reader:
            total_rows += 1
            left_idx = int(row['left_idx'])
            right_idx = int(row['right_idx'])
            if left_idx <= 0 or right_idx <= 0:
                continue

            mkpts0.append([float(row['x1']), float(row['y1'])])
            mkpts1.append([float(row['x2']), float(row['y2'])])
            scores.append(float(row['score']))

    return {
        'mkpts0': np.asarray(mkpts0, dtype=np.float32).reshape(-1, 2),
        'mkpts1': np.asarray(mkpts1, dtype=np.float32).reshape(-1, 2),
        'score': np.asarray(scores, dtype=np.float64).reshape(-1),
        'total_rows': total_rows,
    }


def sort_match_bundle_by_score(match_bundle):
    scores = np.asarray(match_bundle['score'], dtype=np.float64).reshape(-1)
    if scores.size <= 1:
        return match_bundle

    order = np.argsort(-scores, kind='stable')
    if np.array_equal(order, np.arange(scores.size)):
        return match_bundle

    sorted_bundle = dict(match_bundle)
    for key in ('mkpts0', 'mkpts1', 'score'):
        sorted_bundle[key] = np.asarray(match_bundle[key])[order]
    return sorted_bundle


def write_pair_results_csv(output_path, rows):
    fieldnames = [
        'pair_id',
        'csv_file',
        'name0',
        'name1',
        'rot0',
        'rot1',
        'num_matches',
        'num_inliers',
        'num_correct',
        'precision',
        'error_t',
        'error_R',
        'pose_error',
        'estimated',
    ]
    with output_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Estimate Essential matrix from pre-generated CSV matches and evaluate pose.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--input_pairs',
        type=str,
        default=str(DEFAULT_SCANNET_PAIRS),
        help='Path to the list of image pairs with intrinsics and ground truth pose.')
    parser.add_argument(
        '--input_dir',
        type=str,
        default=str(DATA_ROOT),
        help='Path to the directory that contains the images.')
    parser.add_argument(
        '--input_csv_dir',
        type=str,
        required=True,
        help='Directory containing generated CSV files. Search is recursive.')
    parser.add_argument(
        '--output_dir',
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help='Directory to write evaluation_pairs.csv and summary files.')
    parser.add_argument(
        '--max_length',
        type=int,
        default=-1,
        help='Maximum number of pairs to evaluate.')
    parser.add_argument(
        '--resize',
        type=int,
        nargs='+',
        default=[640, 480],
        help='Resize used when the CSV was generated. Must match the CSV source pipeline.')
    parser.add_argument(
        '--resize_float',
        action='store_true',
        help='Use the same resize_float setting as the CSV source pipeline.')
    parser.add_argument(
        '--shuffle',
        action='store_true',
        help='Shuffle ordering of pairs before processing.')
    parser.add_argument(
        '--cache',
        action='store_true',
        help='Skip the run if evaluation_pairs.csv and evaluation_results.* already exist.')
    parser.add_argument(
        '--ransac_threshold',
        type=float,
        default=1.0,
        help='Pose estimation threshold in pixels relative to resized image size.')
    parser.add_argument(
        '--ransac_confidence',
        type=float,
        default=0.99999,
        help='Confidence passed to cv2.findEssentialMat.')
    parser.add_argument(
        '--ransac_max_iters',
        type=int,
        default=1000,
        help='Maximum iteration count passed to cv2.findEssentialMat.')
    parser.add_argument(
        '--usac_method',
        type=str,
        default='USAC_DEFAULT',
        choices=('USAC_DEFAULT', 'USAC_PROSAC'),
        help='OpenCV USAC method passed to cv2.findEssentialMat.')
    parser.add_argument(
        '--epipolar_threshold',
        type=float,
        default=5e-4,
        help='Epipolar error threshold used to mark a correspondence as correct.')
    return parser.parse_args()


def main():
    opt = parse_args()
    print(opt)

    try:
        (
            compute_epipolar_error,
            compute_pose_error,
            pose_auc,
            read_image,
            rotate_intrinsics,
            rotate_pose_inplane,
            scale_intrinsics,
        ) = load_superglue_utils()
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            'Failed to import SuperGlue utilities from third_party/Superglue: {}'.format(exc)
        ) from exc

    if len(opt.resize) == 2 and opt.resize[1] == -1:
        opt.resize = opt.resize[0:1]
    if len(opt.resize) == 2:
        print('Will resize to {}x{} (WxH)'.format(opt.resize[0], opt.resize[1]))
    elif len(opt.resize) == 1 and opt.resize[0] > 0:
        print('Will resize max dimension to {}'.format(opt.resize[0]))
    elif len(opt.resize) == 1:
        print('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')

    input_pairs_path = Path(opt.input_pairs)
    with input_pairs_path.open('r') as handle:
        pairs = [line.split() for line in handle.readlines()]

    if opt.max_length > -1:
        pairs = pairs[0:np.min([len(pairs), opt.max_length])]

    if opt.shuffle:
        random.Random(0).shuffle(pairs)

    if not all(len(pair) == 38 for pair in pairs):
        raise ValueError(
            'All pairs should have ground truth info for evaluation. '
            'File "{}" needs 38 valid entries per row.'.format(input_pairs_path))

    input_dir = Path(opt.input_dir)
    input_csv_dir = Path(opt.input_csv_dir)
    output_dir = Path(opt.output_dir)
    if not input_dir.exists():
        raise FileNotFoundError('input_dir not found: {}'.format(input_dir))
    if not input_csv_dir.exists():
        raise FileNotFoundError('input_csv_dir not found: {}'.format(input_csv_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_csv_path = output_dir / 'evaluation_pairs.csv'
    summary_txt_path = output_dir / 'evaluation_results.txt'
    summary_json_path = output_dir / 'evaluation_results.json'

    if opt.cache and all(path.exists() for path in (
            pair_csv_path, summary_txt_path, summary_json_path)):
        print('Skip evaluation because output files already exist in "{}"'.format(output_dir))
        return

    csv_index = build_csv_index(input_csv_dir)
    if not csv_index:
        raise FileNotFoundError('No CSV files found in {}'.format(input_csv_dir))

    print('Looking for data in directory "{}"'.format(input_dir))
    print('Looking for CSV files in directory "{}"'.format(input_csv_dir))
    print('Will write evaluation results to directory "{}"'.format(output_dir))
    print('Pose estimation backend: cv2.{}'.format(opt.usac_method))

    device = 'cpu'
    progress = ProgressBar(len(pairs))
    pose_errors = []
    precisions = []
    pair_rows = []

    try:
        for i, pair in enumerate(pairs):
            name0, name1 = pair[:2]
            rot0, rot1 = int(pair[2]), int(pair[3])
            pair_id = pair_output_id(name0, name1)
            try:
                csv_path = resolve_csv_path(input_csv_dir, csv_index, pair_id)
                image0, _, scales0 = read_image(
                    input_dir / name0, device, opt.resize, rot0, opt.resize_float)
                image1, _, scales1 = read_image(
                    input_dir / name1, device, opt.resize, rot1, opt.resize_float)
                if image0 is None or image1 is None:
                    raise FileNotFoundError(f'Problem reading image pair: {name0} {name1}')

                match_bundle = sort_match_bundle_by_score(load_csv_matches(csv_path))
                mkpts0 = match_bundle['mkpts0']
                mkpts1 = match_bundle['mkpts1']
            except FileNotFoundError as e:
                print(f"Skipping pair {pair_id} due to missing files: {e}")
                pose_errors.append(np.inf)
                precisions.append(0.0)
                pair_rows.append({
                    'pair_id': pair_id, 'csv_file': 'MISSING',
                    'name0': name0, 'name1': name1,
                    'rot0': rot0, 'rot1': rot1,
                    'num_matches': 0, 'estimated': False,
                    'num_inliers': 0, 'precision': 0.0, 'pose_error': np.inf,
                })
                progress.update(i + 1, 'MISSING')
                continue
            num_matches = len(mkpts0)

            K0 = np.array(pair[4:13]).astype(float).reshape(3, 3)
            K1 = np.array(pair[13:22]).astype(float).reshape(3, 3)
            T_0to1 = np.array(pair[22:]).astype(float).reshape(4, 4)

            K0 = scale_intrinsics(K0, scales0)
            K1 = scale_intrinsics(K1, scales1)

            if rot0 != 0 or rot1 != 0:
                cam0_T_w = np.eye(4)
                cam1_T_w = T_0to1
                if rot0 != 0:
                    K0 = rotate_intrinsics(K0, image0.shape, rot0)
                    cam0_T_w = rotate_pose_inplane(cam0_T_w, rot0)
                if rot1 != 0:
                    K1 = rotate_intrinsics(K1, image1.shape, rot1)
                    cam1_T_w = rotate_pose_inplane(cam1_T_w, rot1)
                T_0to1 = cam1_T_w @ np.linalg.inv(cam0_T_w)

            if len(mkpts0) > 0:
                epi_errs = compute_epipolar_error(mkpts0, mkpts1, T_0to1, K0, K1)
                correct = epi_errs < opt.epipolar_threshold
            else:
                correct = np.zeros((0,), dtype=bool)

            num_correct = int(np.sum(correct))
            precision = float(np.mean(correct)) if len(correct) > 0 else 0.0

            pose_ret = estimate_pose_usac(
                mkpts0,
                mkpts1,
                K0,
                K1,
                opt.ransac_threshold,
                conf=opt.ransac_confidence,
                max_iters=opt.ransac_max_iters,
                method_name=opt.usac_method,
                return_pose_data=True,
            )
            if pose_ret is None:
                estimated = False
                err_t, err_R = np.inf, np.inf
                num_inliers = 0
            else:
                estimated = True
                R = pose_ret['R']
                num_inliers = pose_ret['num_inliers']
                err_t, err_R = compute_pose_error(T_0to1, R, pose_ret['t'])

            pose_error = float(np.maximum(err_t, err_R))
            pose_errors.append(pose_error)
            precisions.append(precision)
            pair_rows.append({
                'pair_id': pair_id,
                'csv_file': str(csv_path.name),
                'name0': name0,
                'name1': name1,
                'rot0': rot0,
                'rot1': rot1,
                'num_matches': num_matches,
                'num_inliers': num_inliers,
                'num_correct': num_correct,
                'precision': precision,
                'error_t': err_t,
                'error_R': err_R,
                'pose_error': pose_error,
                'estimated': estimated,
            })
            progress.update(i + 1, extra=pair_id)
    finally:
        progress.close()

    write_pair_results_csv(pair_csv_path, pair_rows)

    aucs = pose_auc(pose_errors, [5, 10, 20])
    aucs = [100. * value for value in aucs]
    prec = 100. * np.mean(precisions)
    summary_text = save_evaluation_summary(output_dir, len(pose_errors), aucs, prec)
    print(summary_text, end='')


if __name__ == '__main__':
    main()
