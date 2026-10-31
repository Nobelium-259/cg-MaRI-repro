"""Course inference adapter with explicit inputs and synchronized timing."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import time
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_MARI_SHA256 = '6d7bf5bee85ab3e113ad729f7dfe374285e818693da24e8276c72dd0bd64fb0f'


def read_manifest(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(data, dict):
        raise ValueError('manifest must be a JSON object')
    for key in ('gallery', 'queries'):
        items = data.get(key)
        if not isinstance(items, list) or not items:
            raise ValueError(f'{key} must be a nonempty list')
        seen = set()
        for item in items:
            if not isinstance(item, dict):
                raise ValueError(f'each {key} item must be an object')
            identifier = item.get('id')
            if not isinstance(identifier, str) or not identifier.strip():
                raise ValueError(f'{key} item requires a nonempty string id')
            if identifier in seen:
                raise ValueError(f'duplicate {key} id: {identifier}')
            seen.add(identifier)
            for field in ('path', 'mask'):
                value = item.get(field)
                if field == 'mask' and value is None:
                    continue
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f'{key} {identifier}: {field} must be a nonempty path')
                resolved = (path.parent / value).resolve()
                if not resolved.is_file():
                    raise FileNotFoundError(f'{key} {identifier}: missing {field}: {resolved}')
                item[field] = str(resolved)
    known = {item['id'] for item in data['gallery']}
    for query in data['queries']:
        relevant = query.get('relevant_ids', [])
        if not isinstance(relevant, list) or any(not isinstance(x, str) for x in relevant):
            raise ValueError(f"query {query['id']}: relevant_ids must be a list of strings")
        if len(relevant) != len(set(relevant)):
            raise ValueError(f"query {query['id']}: duplicate relevance IDs")
        if set(relevant) - known:
            raise ValueError(f"query {query['id']}: unknown relevance ID")
    return data


def rank(features, query, k=5):
    features = np.asarray(features, dtype=np.float32)
    query = np.asarray(query, dtype=np.float32)
    if features.ndim != 2 or not all(features.shape) or query.shape != (features.shape[1],):
        raise ValueError('feature shape mismatch or empty gallery')
    if not np.isfinite(features).all() or not np.isfinite(query).all():
        raise ValueError('features must contain only finite values (no NaN/Inf)')
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError('k must be a positive integer')
    # Float64 norms prevent overflow for otherwise finite float32 inputs.
    features = features / np.maximum(np.linalg.norm(features.astype(np.float64), axis=1, keepdims=True), 1e-12)
    query = query / max(np.linalg.norm(query.astype(np.float64)), 1e-12)
    scores = np.clip(features @ query, -1, 1)
    order = np.argsort(-scores, kind='stable')[:min(k, len(scores))]
    return order, scores[order]


def metrics(predictions, queries):
    hit1, hit5, average_precision = [], [], []
    for query in queries:
        relevant = set(query.get('relevant_ids', []))
        if not relevant:
            continue
        order = predictions[query['id']]
        if len(order) != len(set(order)):
            raise ValueError(f"duplicate predictions for query {query['id']}")
        hit1.append(bool(set(order[:1]) & relevant))
        hit5.append(bool(set(order[:5]) & relevant))
        count, value = 0, 0
        for position, identifier in enumerate(order, 1):
            if identifier in relevant:
                count += 1
                value += count / position
        average_precision.append(value / len(relevant))
    if not average_precision:
        raise ValueError('no labeled queries; use the interface for qualitative retrieval')
    return {'queries': len(average_precision), 'hit_at_1': float(np.mean(hit1)),
            'hit_at_5': float(np.mean(hit5)), 'mAP': float(np.mean(average_precision))}


def validate_mask_geometry(image_size, mask_size):
    # Official MaRI pairs use a higher-resolution mask in the same coordinates.
    # Independent Resize/CenterCrop (as upstream) preserves matching aspect ratios.
    # Permit at most 0.1% relative aspect drift from source export rounding.
    image_ratio = image_size[0] / image_size[1]
    mask_ratio = mask_size[0] / mask_size[1]
    if abs(image_ratio - mask_ratio) > 0.001 * max(image_ratio, mask_ratio):
        raise ValueError(f'mask/image aspect ratios differ: {mask_size} vs {image_size}')


def load_image(path, mode):
    try:
        with Image.open(path) as image:
            return image.convert(mode)
    except (OSError, ValueError) as error:
        raise ValueError(f'cannot read image {path}: {error}') from error


class Descriptor:
    def encode(self, path, branch='image', mask=None):
        image = load_image(path, 'RGB')
        array = np.asarray(ImageOps.fit(image, (64, 64)), dtype=np.float32) / 255
        histogram = np.concatenate([np.histogram(array[:, :, i], bins=16, range=(0, 1))[0]
                                    for i in range(3)]).astype('float32')
        histogram /= max(np.linalg.norm(histogram), 1e-12)
        gray = array.mean(2)
        gradient = np.concatenate([np.diff(gray, axis=0).ravel(), np.diff(gray, axis=1).ravel()])
        texture = np.histogram(gradient, bins=16, range=(-1, 1))[0].astype('float32')
        texture /= max(np.linalg.norm(texture), 1e-12)
        return np.concatenate([histogram, texture])


def sha256(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class DinoEncoder:
    def __init__(self, mode, checkpoint, repo, device='cpu'):
        import copy
        import torch
        from torchvision import transforms
        if mode not in ('mari', 'dinov2'):
            raise ValueError(f'unknown neural model: {mode}')
        self.torch = torch
        if not os.environ.get('TORCH_HOME'):
            torch.hub.set_dir(str(ROOT / '.cache/torch/hub'))
        self.device = torch.device(device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but torch.cuda.is_available() is false')
        torch.set_num_threads(min(4, torch.get_num_threads()))
        if mode == 'mari' and (not checkpoint or not Path(checkpoint).is_file()):
            raise FileNotFoundError('MaRI weights required; no baseline fallback')
        if repo and not (Path(repo) / 'hubconf.py').is_file():
            raise FileNotFoundError(f'local DINOv2 hubconf.py not found: {repo}')
        self.checkpoint_sha256 = sha256(checkpoint) if mode == 'mari' else None
        if mode == 'mari' and self.checkpoint_sha256 != EXPECTED_MARI_SHA256:
            raise ValueError('MaRI checkpoint SHA256 mismatch; official course snapshot required')
        self.image = torch.hub.load(str(repo) if repo else 'facebookresearch/dinov2',
                                    'dinov2_vitl14', source='local' if repo else 'github',
                                    pretrained=mode == 'dinov2', trust_repo=True)
        self.material = self.image
        if mode == 'mari':
            state = torch.load(str(checkpoint), map_location='cpu', weights_only=True)
            state = state.get('model_state_dict', state)
            if not isinstance(state, dict):
                raise ValueError('checkpoint must contain a model state dictionary')
            state = {key.removeprefix('module.'): value for key, value in state.items()}
            self.material = copy.deepcopy(self.image)
            for prefix, model in [('render_encoder.', self.image), ('material_encoder.', self.material)]:
                model.load_state_dict({key[len(prefix):]: value for key, value in state.items()
                                       if key.startswith(prefix)}, strict=True)
            del state
        self.image.eval().to(self.device)
        self.material.eval().to(self.device)
        self.transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
            transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        self.mask_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                                                  transforms.ToTensor()])

    def encode(self, path, branch='image', mask=None):
        if branch not in ('image', 'material'):
            raise ValueError('branch must be image or material')
        image = load_image(path, 'RGB')
        tensor = self.transform(image).unsqueeze(0)
        if mask:
            mask_image = load_image(mask, 'L')
            validate_mask_geometry(image.size, mask_image.size)
            tensor = tensor * self.mask_transform(mask_image).unsqueeze(0)
        model = self.material if branch == 'material' else self.image
        with self.torch.inference_mode():
            feature = model(tensor.to(self.device))
            feature = self.torch.nn.functional.normalize(feature, dim=-1)
        feature = feature.cpu().numpy()[0]
        if not np.isfinite(feature).all():
            raise ValueError(f'model returned NaN/Inf features: {path}')
        return feature

    def synchronize(self):
        if self.device.type == 'cuda':
            self.torch.cuda.synchronize(self.device)


def make_encoder(mode, checkpoint=None, repo=None, device='cpu'):
    if mode == 'baseline':
        return Descriptor()
    if mode not in ('mari', 'dinov2'):
        raise ValueError(f'unknown model: {mode}; no baseline fallback')
    return DinoEncoder(mode, checkpoint, repo, device)


def synchronize(encoder):
    if hasattr(encoder, 'synchronize'):
        encoder.synchronize()


def run(manifest, mode='baseline', checkpoint=None, repo=None, device='cpu', output=None, warmup=2):
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError('warmup must be a nonnegative integer')
    data = read_manifest(manifest)
    start = time.perf_counter()
    encoder = make_encoder(mode, checkpoint, repo, device)
    synchronize(encoder)
    model_load_seconds = time.perf_counter() - start
    start = time.perf_counter()
    for _ in range(warmup):
        encoder.encode(data['gallery'][0]['path'], 'material')
        query = data['queries'][0]
        encoder.encode(query['path'], 'image', query.get('mask'))
    synchronize(encoder)
    warmup_seconds = time.perf_counter() - start
    synchronize(encoder)
    start = time.perf_counter()
    gallery = np.stack([encoder.encode(item['path'], 'material') for item in data['gallery']])
    synchronize(encoder)
    index_seconds = time.perf_counter() - start
    predictions, records = {}, []
    for query in data['queries']:
        synchronize(encoder)
        start = time.perf_counter()
        feature = encoder.encode(query['path'], 'image', query.get('mask'))
        order, scores = rank(gallery, feature, len(gallery))
        synchronize(encoder)
        latency = time.perf_counter() - start
        identifiers = [data['gallery'][i]['id'] for i in order]
        predictions[query['id']] = identifiers
        records.append({'query_id': query['id'], 'latency_seconds': latency,
                        'ranking': [{'id': identifier, 'cosine': float(score)}
                                    for identifier, score in zip(identifiers, scores)]})
    result = {'mode': mode, 'device': str(encoder.device) if hasattr(encoder, 'device') else 'cpu',
        'requested_device': device, 'run_at_utc': datetime.now(timezone.utc).isoformat(),
        'protocol': data.get('protocol'), 'warning': data.get('warning'),
        'gallery_size': len(gallery), 'model_load_seconds': model_load_seconds,
        'warmup_iterations_per_branch': warmup, 'warmup_seconds': warmup_seconds,
        'index_seconds': index_seconds, 'query_median_seconds': float(np.median([r['latency_seconds'] for r in records])),
        'timing_protocol': 'perf_counter with explicit CUDA synchronization before/after each stage; '
                           'queries include image IO, preprocessing, device transfers, forward, normalization and ranking; '
                           'FP32, batch=1; MaRI model load includes task-checkpoint SHA256; gallery built once after warmup; '
                           'single pass, shared desktop GPU, not an isolated performance benchmark',
        'manifest_sha256': sha256(manifest), 'metrics': metrics(predictions, data['queries']), 'records': records}
    if mode == 'mari':
        result['checkpoint_sha256'] = encoder.checkpoint_sha256
    if hasattr(encoder, 'torch'):
        result['torch_version'] = encoder.torch.__version__
        result['cuda_runtime'] = encoder.torch.version.cuda
        result['gpu_name'] = encoder.torch.cuda.get_device_name(encoder.device) if encoder.device.type == 'cuda' else None
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    return result
