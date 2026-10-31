"""Prepare pinned public CC0 demo assets without overwriting a local course manifest."""
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.parse import urlparse
from io import BytesIO
import argparse,hashlib,json,shutil,re
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
HOSTS={'acg-media.struffelproductions.com','f003.backblazeb2.com'}
def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-local',type=Path,help='Reuse the matching asset_probe_20261031 PNG files, without network access')
    args=parser.parse_args()
    sources=json.loads((ROOT/'data/public_demo_sources.json').read_text(encoding='utf8'))
    out=ROOT/'data/public_demo';out.mkdir(parents=True,exist_ok=True)
    record_path=out/'download_record.json'
    previous=json.loads(record_path.read_text(encoding='utf8')) if record_path.exists() else []
    verified={(r['id'],r['branch']):r for r in previous}
    gallery=[];queries=[];records=[]
    for asset in sources['assets']:
        aid=asset['id']
        if not re.fullmatch(r'[A-Za-z0-9_]+',aid):raise ValueError('Invalid asset ID')
        for branch in ['gallery','query']:
            source=asset[branch];target=out/(aid+'_'+branch+'.png')
            if target.exists():
                actual=digest(target);known=verified.get((aid,branch),{})
                # PNG encoding can differ by Pillow version; accept this tool's own
                # verified download record as well as the original snapshot bytes.
                recorded=known.get('url')==source['url'] and known.get('sha256')==actual
                if actual!=source['saved_sha256'] and not recorded:raise RuntimeError('Existing file differs; preserved: '+str(target))
            elif args.from_local:
                original=args.from_local/target.name
                if digest(original)!=source['saved_sha256']:raise RuntimeError('Local asset hash mismatch: '+str(original))
                shutil.copy2(original,target)
            else:
                url=source['url'];parsed=urlparse(url)
                if parsed.scheme!='https' or parsed.hostname not in HOSTS:raise ValueError('Unexpected asset host')
                with urlopen(Request(url,headers={'User-Agent':'MaRI-repro-demo/1.0'}),timeout=30) as response:
                    payload=response.read(5*1024*1024+1)
                if len(payload)>5*1024*1024 or hashlib.sha256(payload).hexdigest()!=source['download_sha256']:
                    raise RuntimeError('Official preview content changed or was truncated: '+aid)
                with Image.open(BytesIO(payload)) as image:
                    if image.size!=tuple(source['size']):raise ValueError('Unexpected image size')
                    if image.mode=='RGBA':
                        background=Image.new('RGBA',image.size,'white');background.alpha_composite(image);image=background.convert('RGB')
                    else:image=image.convert('RGB')
                    with target.open('xb') as stream:image.save(stream,format='PNG')
            records.append({'id':aid,'branch':branch,'url':source['url'],'sha256':digest(target)})
        gallery.append({'id':aid,'path':aid+'_gallery.png','category':asset['category']})
        queries.append({'id':aid,'path':aid+'_query.png','relevant_ids':[aid]})
    manifest={'protocol':'ambientcg_flat_texture_to_preview_probe_v1','warning':'12 public CC0 asset pairs, flat color texture to material preview. Not real-world photos or the original paper benchmark; training overlap unverified.','gallery':gallery,'queries':queries}
    path=out/'manifest.json';path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    (out/'download_record.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf8')
    default=ROOT/'data/demo_manifest.json'
    if not default.exists():
        for group in ['gallery','queries']:
            for item in manifest[group]:item['path']='public_demo/'+item['path']
        with default.open('x',encoding='utf8') as stream:json.dump(manifest,stream,ensure_ascii=False,indent=2)
        print('Default demo manifest created:',default)
    else:print('Existing default manifest preserved. For this dataset use --manifest',path)
    print('Prepared',len(queries),'queries and',len(gallery),'gallery assets')
if __name__=='__main__':main()
