"""Private bounded source-ordered feature export; no model or readout fitting.

Caller owns source/checkpoint admission, the shared compute lock and FD silence.
Returned per-image embeddings are private and must never be emitted to a model.
"""
from concurrent.futures import ThreadPoolExecutor
import os

import numpy as np
import torch

ARMS = ('base', 'source_control', 'multisource_candidate')
ERROR = 'multisource retinal export failed'


def require(value):
    if not value:
        raise ValueError(ERROR) from None


def image_batch(sources, source, indices):
    """Authenticated reader → fixed224 RGB normalization, without augmentation."""
    require(source in ('brset', 'odir'))
    require(type(indices) is np.ndarray and indices.dtype == np.int64 and indices.ndim == 1
            and len(indices) > 0 and np.all(indices >= 0))

    def one(index):
        value = sources.read(source, int(index))
        require(type(value) is np.ndarray and value.dtype == np.uint8 and value.shape == (224, 224, 3))
        return value.copy()

    with ThreadPoolExecutor(max_workers=4) as pool:
        pixels = np.stack(list(pool.map(one, indices)))
    value = pixels.astype(np.float32) / np.float32(255)
    value = (value - np.asarray([.485, .456, .406], np.float32)) / np.asarray([.229, .224, .225], np.float32)
    return torch.from_numpy(np.ascontiguousarray(value.transpose(0, 3, 1, 2)))


def export_arm(encoder, sources, *, device='mps', batch_size=16):
    """Export exact source order at384 dimensions, excluding five prefix tokens.

The encoder must enter in evaluation mode; no dropout/stochastic training path.
Inference is no-grad. Its original device is restored before return or failure.
CPU is available explicitly for synthetic tests; MPS never falls back silently.
"""
    original_device = None
    try:
        require(isinstance(encoder, torch.nn.Module) and not encoder.training)
        require(type(batch_size) is int and 0 < batch_size <= 16)
        destination = torch.device(device)
        require(destination.type in ('cpu', 'mps'))
        require(destination.type != 'mps' or torch.backends.mps.is_available())
        require(destination.type != 'mps' or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') == '0')
        parameters = list(encoder.parameters())
        require(parameters)
        devices = {value.device for value in parameters}
        require(len(devices) == 1)
        original_device = parameters[0].device
        encoder.to(destination)
        result = {}
        for source in ('brset', 'odir'):
            n = len(sources.private_specs[source]['image_groups'])
            require(n > 0)
            features = np.empty((n, 384), np.float32)
            for start in range(0, n, batch_size):
                indices = np.arange(start, min(start + batch_size, n), dtype=np.int64)
                images = image_batch(sources, source, indices).to(destination)
                with torch.inference_mode():
                    if callable(getattr(encoder, 'encode_student', None)):
                        z = encoder.encode_student(images)
                    else:
                        require(getattr(encoder, 'num_prefix_tokens', None) == 5
                                and getattr(encoder, 'embed_dim', None) == 384)
                        tokens = encoder.forward_features(images)
                        require(tuple(tokens.shape) == (len(indices), 201, 384))
                        z = tokens[:, 5:].mean(1)
                    require(tuple(z.shape) == (len(indices), 384) and torch.isfinite(z).all().item())
                    features[indices] = z.float().cpu().numpy()
            result[source] = features
        return result
    except Exception:
        raise ValueError(ERROR) from None
    finally:
        if original_device is not None:
            encoder.to(original_device)


def arrange_arms(by_arm):
    """Transpose private arm exports to the exact evaluation source/arm schema."""
    try:
        require(type(by_arm) is dict and set(by_arm) == set(ARMS))
        require(all(type(value) is dict and set(value) == {'brset', 'odir'} for value in by_arm.values()))
        result = {source: {} for source in ('brset', 'odir')}
        for source in result:
            expected = by_arm['base'][source].shape
            require(len(expected) == 2 and expected[0] > 0 and expected[1] == 384)
            for arm in ARMS:
                value = by_arm[arm][source]
                require(type(value) is np.ndarray and value.dtype == np.float32 and value.shape == expected
                        and np.isfinite(value).all())
                result[source][arm] = value
        return result
    except Exception:
        raise ValueError(ERROR) from None
