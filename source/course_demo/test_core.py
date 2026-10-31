import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from core import Descriptor, make_encoder, metrics, rank, read_manifest
from runtime import RetrievalRuntime


class Tests(unittest.TestCase):
    def test_small_gallery_and_zero(self):
        self.assertEqual(rank([[1, 0], [0, 1]], [0, 1])[0].tolist(), [1, 0])
        self.assertEqual(rank([[0, 0]], [0, 0])[0].tolist(), [0])

    def test_multiple_relevant(self):
        result = metrics({'q': ['a', 'x', 'b']}, [{'id': 'q', 'relevant_ids': ['a', 'b']}])
        self.assertAlmostEqual(result['mAP'], (1 + 2 / 3) / 2)

    def test_unlabeled(self):
        with self.assertRaisesRegex(ValueError, 'no labeled'):
            metrics({'q': ['a']}, [{'id': 'q'}])

    def test_duplicate_predictions(self):
        with self.assertRaisesRegex(ValueError, 'duplicate predictions'):
            metrics({'q': ['a', 'a']}, [{'id': 'q', 'relevant_ids': ['a']}])

    def test_nonfinite_features(self):
        for features, query in [([[np.nan, 0]], [1, 0]), ([[1, 0]], [np.inf, 0])]:
            with self.subTest(features=features, query=query), self.assertRaisesRegex(ValueError, 'finite'):
                rank(features, query)

    def test_empty_shapes_and_invalid_k(self):
        for features, query, count in [([], [], 5), ([[1, 0]], [1], 5), ([[1, 0]], [1, 0], 0),
                                       ([[1, 0]], [1, 0], True), ([[1, 0]], [1, 0], 1.5)]:
            with self.subTest(count=count), self.assertRaises(ValueError):
                rank(features, query, count)

    def test_large_finite_features(self):
        _, scores = rank([[1e30, 0]], [1e30, 0])
        self.assertAlmostEqual(float(scores[0]), 1)

    def test_unknown_model_never_falls_back(self):
        with self.assertRaisesRegex(ValueError, 'unknown model'):
            make_encoder('typo')

    def test_manifest_boundaries(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'image.png').write_bytes(b'test')
            original = {'gallery': [{'id': 'a', 'path': 'image.png'}],
                        'queries': [{'id': 'q', 'path': 'image.png', 'relevant_ids': ['a']}]}
            cases = []
            for key in ('queries', 'gallery'):
                empty = copy.deepcopy(original)
                empty[key] = []
                cases.append((empty, 'nonempty'))
                duplicate = copy.deepcopy(original)
                duplicate[key].append(copy.deepcopy(duplicate[key][0]))
                cases.append((duplicate, 'duplicate'))
            missing = copy.deepcopy(original)
            del missing['queries'][0]['path']
            cases.append((missing, 'path'))
            wrong = copy.deepcopy(original)
            wrong['queries'][0]['relevant_ids'] = 'a'
            cases.append((wrong, 'list of strings'))
            unknown = copy.deepcopy(original)
            unknown['queries'][0]['relevant_ids'] = ['unknown']
            cases.append((unknown, 'unknown relevance'))
            for content, message in cases:
                (root / 'manifest.json').write_text(json.dumps(content), encoding='utf8')
                with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                    read_manifest(root / 'manifest.json')
            (root / 'manifest.json').write_text(json.dumps(original), encoding='utf8')
            self.assertEqual(read_manifest(root / 'manifest.json')['gallery'][0]['id'], 'a')

    def test_missing_mask(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'image.png').write_bytes(b'test')
            content = {'gallery': [{'id': 'a', 'path': 'image.png'}],
                       'queries': [{'id': 'q', 'path': 'image.png', 'mask': 'missing.png'}]}
            (root / 'manifest.json').write_text(json.dumps(content), encoding='utf8')
            with self.assertRaisesRegex(FileNotFoundError, 'missing mask'):
                read_manifest(root / 'manifest.json')

    def test_bad_image(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'bad.png'
            path.write_bytes(b'not an image')
            with self.assertRaisesRegex(ValueError, 'cannot read image'):
                Descriptor().encode(path)

    def test_runtime_single_model_and_same_path_content_update(self):
        import os
        class FakeEncoder:
            def encode(self, path, branch='image', mask=None):
                return np.array([Path(path).read_bytes()[0], 1], dtype=np.float32)
        calls = []
        def factory(*arguments):
            calls.append(arguments)
            return FakeEncoder()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'image.png'
            path.write_bytes(b'a')
            items = [{'id': 'a', 'path': str(path)}]
            runtime = RetrievalRuntime(factory)
            runtime.retrieve('baseline', None, None, 'cpu', items, path)
            runtime.retrieve('baseline', None, None, 'cpu', items, path)
            self.assertEqual((runtime.loads, runtime.index_builds), (1, 1))
            before = path.stat()
            path.write_bytes(b'b')
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
            runtime.retrieve('baseline', None, None, 'cpu', items, path)
            self.assertEqual((runtime.loads, runtime.index_builds), (1, 2))
            runtime.retrieve('dinov2', None, None, 'cpu', items, path)
            self.assertEqual((runtime.loads, runtime.index_builds), (2, 3))
            runtime.clear()
            self.assertIsNone(runtime.encoder)
            self.assertIsNone(runtime.gallery)


if __name__ == '__main__':
    unittest.main(verbosity=2)
