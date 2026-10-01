"""Private source adapters and pixel batching; no discovery or training admission.

Inputs must come from separately authenticated local qualification artifacts.
Callers own FD-silencing, the shared compute lock and original-source approval.
No patient values, IDs or source-local indices may leave the local process.
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

import bran_retinal_adaptation_evaluation_v1 as reference

ERROR = 'invalid multisource retinal source contract'


def require(value):
    if not value:
        raise ValueError(ERROR) from None


def brset_spec(arrays):
    """Preserve admitted BRSET order, eye labels and old train-only weight rule."""
    try:
        q = arrays
        groups, split = q['groups'], q['split']
        require(type(groups) is np.ndarray and groups.dtype == np.int64 and groups.ndim == 1)
        n = len(groups)
        require(n > 0 and np.all(groups >= 0) and np.max(groups) < n)
        unique = np.unique(groups)
        require(np.array_equal(unique, np.arange(len(unique))))
        patient = reference.group_patients(groups, split, q['labels'], q['observed'], q['ages'],
            {arm: np.zeros((n, 1), np.float32) for arm in reference.ARMS})
        weights, usable = reference.training_label_policy(q['labels'], q['observed'], split, patient)
        spec = {'image_groups': groups.copy(), 'group_split': patient['split'].astype(np.uint8),
                'labels': q['labels'].copy(), 'observed': q['observed'].copy(), 'usable': usable.copy()}
        return spec, weights.copy(), np.arange(n, dtype=np.int64)
    except Exception:
        raise ValueError(ERROR) from None


def odir_spec(inputs, split, selection, retained):
    """Join retained actual eyes to the already-frozen ODIR patient split.

No new split, imputation, demographic matching or per-eye duplication of labels.
Returned image indices address the original qualified content selection.
"""
    try:
        require(type(inputs) is dict and set(inputs) == {'patient_ids', 'labels', 'eye_counts'})
        ids, labels, counts = (inputs[k] for k in ('patient_ids', 'labels', 'eye_counts'))
        require(type(ids) is np.ndarray and ids.ndim == 1 and ids.dtype.kind == 'U' and len(ids) > 0)
        p = len(ids)
        require(all(len(value) > 0 for value in ids) and len(set(ids.tolist())) == p)
        require(type(labels) is np.ndarray and labels.shape == (p, 8) and labels.dtype.kind == 'f')
        require(np.all(np.isnan(labels) | np.isin(labels, (0., 1.))))
        require(type(counts) is np.ndarray and counts.dtype == np.int64 and counts.shape == (p,)
                and np.all(np.isin(counts, (1, 2))))
        require(type(split) is np.ndarray and split.dtype == np.uint8 and split.shape == (p,)
                and np.all(np.isin(split, (0, 1, 2))) and np.any(split == 0))
        require(type(selection) is list and len(selection) > 0)
        require(type(retained) is np.ndarray and retained.dtype == bool and retained.shape == (len(selection),))
        lookup = {value: index for index, value in enumerate(ids.tolist())}
        rows = np.flatnonzero(retained).astype(np.int64)
        group = []
        eye_seen = set()
        for index in rows:
            item = selection[index]
            require(type(item) is dict and set(item) == {'patient_id', 'eye', 'member'})
            require(type(item['patient_id']) is str and item['patient_id'] in lookup)
            require(item['eye'] in ('left', 'right') and type(item['member']) is str and item['member'])
            key = (item['patient_id'], item['eye'])
            require(key not in eye_seen)
            eye_seen.add(key)
            group.append(lookup[item['patient_id']])
        group = np.asarray(group, dtype=np.int64)
        require(np.array_equal(np.bincount(group, minlength=p), counts))
        observed = np.isfinite(labels)
        train = split == 0
        support = np.asarray([[np.sum(train & observed[:, j] & (labels[:, j] == value))
                               for value in (0, 1)] for j in range(8)])
        usable = np.all(support >= 20, axis=1)
        weights = np.ones(8, np.float32)
        weights[usable] = np.clip(np.sqrt(support[usable, 0] / support[usable, 1]), 1, 10)
        spec = {'image_groups': group, 'group_split': split.copy(), 'labels': labels.copy(),
                'observed': observed, 'usable': usable}
        return spec, weights, rows
    except Exception:
        raise ValueError(ERROR) from None


class PixelBridge:
    """Turn private planner selections into model batches with bounded I/O.

Each source reader receives a source-local image index and MUST authenticate
raw/decoded bytes before returning uint8 RGB224 pixels. This class does not
certify the readers or grant permission to use a source. It does not cache data.
"""

    def __repr__(self):
        return '<PixelBridge private>'

    def __init__(self, readers):
        require(type(readers) is dict and set(readers) == {'brset', 'odir'}
                and all(callable(v) for v in readers.values()))
        self._readers = readers.copy()

    def materialize(self, plan):
        try:
            require(type(plan) is dict and set(plan) == {'brset', 'odir'})
            output = {}
            for source in ('brset', 'odir'):
                value = plan[source]
                require(type(value) is dict and set(value) == {
                    'image_indices', 'flips', 'patch_mask', 'group_index', 'labels', 'observed'})
                indices, flips = value['image_indices'], value['flips']
                require(type(indices) is np.ndarray and indices.dtype == np.int64 and indices.ndim == 1
                        and len(indices) > 0 and np.all(indices >= 0))
                require(type(flips) is np.ndarray and flips.dtype == bool and flips.shape == indices.shape)

                def one(index):
                    pixels = self._readers[source](int(index))
                    require(type(pixels) is np.ndarray and pixels.dtype == np.uint8 and pixels.shape == (224, 224, 3))
                    return pixels.copy()

                with ThreadPoolExecutor(max_workers=4) as pool:
                    pixels = np.stack(list(pool.map(one, indices)))
                pixels[flips] = pixels[flips, :, ::-1, :]
                images = pixels.astype(np.float32) / np.float32(255)
                images = (images - np.asarray([.485, .456, .406], np.float32)) / np.asarray([.229, .224, .225], np.float32)
                output[source] = {'images': torch.from_numpy(np.ascontiguousarray(images.transpose(0, 3, 1, 2)))}
                for key in ('patch_mask', 'group_index', 'labels', 'observed'):
                    require(type(value[key]) is np.ndarray)
                    output[source][key] = torch.from_numpy(value[key].copy())
            return output
        except Exception:
            raise ValueError(ERROR) from None
