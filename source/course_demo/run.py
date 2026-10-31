import argparse
import json
from core import ROOT, run

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['baseline', 'mari', 'dinov2'], default='baseline')
parser.add_argument('--manifest', default=str(ROOT / 'data/demo_manifest.json'))
parser.add_argument('--checkpoint', default=str(ROOT / 'weights/mari.pth'))
parser.add_argument('--repo', default=str(ROOT / 'source/dinov2'))
parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
parser.add_argument('--warmup', type=int, default=2, help='warmup forwards per branch, excluded from query timing')
parser.add_argument('--output', default=str(ROOT / 'results/new_run.json'))
if __name__ == '__main__':
    print(json.dumps(run(**vars(parser.parse_args())), ensure_ascii=False, indent=2, allow_nan=False))
