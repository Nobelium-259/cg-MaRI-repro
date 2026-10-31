"""Verify saved comparisons against full rankings and reusable feature banks."""
from pathlib import Path
import json,tempfile,unittest,subprocess,sys
from unittest.mock import patch
import numpy as np
import comparison
from comparison import load_comparison,cases_for,load_bank,records_for,local_path
from core import rank,sha256,ROOT

class ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.bundle=load_comparison()

    def test_all_cases_partition_without_hiding_failures(self):
        for split,summary in self.bundle['summary']['splits'].items():
            all_cases=cases_for(self.bundle,split)
            groups=[cases_for(self.bundle,split,k) for k in ['mari_only','dinov2_only','both_correct','both_wrong']]
            self.assertEqual(len(all_cases),summary['queries'])
            self.assertEqual(sorted(x['query_id'] for g in groups for x in g),sorted(x['query_id'] for x in all_cases))
            for key,group in zip(['mari_only','dinov2_only','both_correct','both_wrong'],groups):self.assertEqual(len(group),summary['paired_outcomes'][key])

    def test_feature_banks_reproduce_all_complete_rankings(self):
        for mode in ['mari','dinov2']:
            with np.load(self.bundle['folder']/(mode+'_features.npz'),allow_pickle=False) as bank:
                for split,manifest in self.bundle['manifests'].items():
                    features=load_bank(self.bundle,mode,split)
                    query_features=bank[split+'_queries']
                    self.assertEqual(len(query_features),len(manifest['queries']))
                    for query,feature in zip(manifest['queries'],query_features):
                        order,scores=rank(features,feature,len(features))
                        record=records_for(self.bundle,split,query['id'])[mode]
                        self.assertEqual([manifest['gallery'][i]['id'] for i in order],[x['id'] for x in record['ranking']])
                        np.testing.assert_allclose(scores,[x['cosine'] for x in record['ranking']],atol=1e-7)

    def test_corrupt_feature_bank_rejected_before_inference(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);(p/'mari_features.npz').write_bytes(b'corrupted')
            meta=json.loads((self.bundle['folder']/'mari_features.json').read_text('utf-8'))
            (p/'mari_features.json').write_text(json.dumps(meta),encoding='utf-8')
            altered=dict(self.bundle,folder=p)
            with self.assertRaisesRegex(ValueError,'checksum'):load_bank(altered,'mari','real_masked')

    def test_completed_run_cannot_be_overwritten(self):
        path=self.bundle['folder']/'baseline.json';before=sha256(path)
        process=subprocess.run([sys.executable,str(ROOT/'source/course_demo/compare_models.py'),'--mode','baseline','--run-id',self.bundle['folder'].name],capture_output=True,text=True,encoding='utf-8')
        self.assertNotEqual(process.returncode,0)
        self.assertIn('Preserve the existing run',process.stderr)
        self.assertEqual(sha256(path),before)
        self.assertFalse((self.bundle['folder']/'baseline.running').exists())

    def test_invalid_filter_and_escaping_path(self):
        with self.assertRaisesRegex(ValueError,'outcome'):cases_for(self.bundle,'real_masked','fake')
        with self.assertRaisesRegex(ValueError,'inside'):local_path('../outside.json')

    def test_live_failure_is_explicit_no_fallback(self):
        # Use an intentional missing checkpoint. No model forward is run.
        q=self.bundle['manifests']['real_masked']['queries'][0]
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(FileNotFoundError,'MaRI weights required'):
                comparison.live_compare(self.bundle,'real_masked',q['path'],q['mask'],Path(temp)/'missing.pth')

    def test_displayed_input_matches_normalize_then_mask(self):
        import torch
        from PIL import Image
        from torchvision import transforms
        torch.set_num_threads(1)
        transform=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
        mask_transform=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor()])
        rng=np.random.default_rng(7)
        with tempfile.TemporaryDirectory() as temp:
            image=Path(temp)/'image.png';mask=Path(temp)/'mask.png'
            Image.fromarray(rng.integers(0,256,(67,105,3),dtype=np.uint8)).save(image)
            Image.fromarray(rng.integers(0,256,(134,210,3),dtype=np.uint8)).save(mask)
            with Image.open(image) as im:x=transform(im.convert('RGB'))
            with Image.open(mask) as im:alpha=mask_transform(im.convert('L'))
            expected=(x*alpha*torch.tensor([.229,.224,.225])[:,None,None]+torch.tensor([.485,.456,.406])[:,None,None]).permute(1,2,0).numpy()
            actual=np.asarray(comparison.input_preview(image,mask))/255
            self.assertLessEqual(float(np.max(np.abs(expected-actual))),.501/255)

if __name__=='__main__':unittest.main(verbosity=2)
