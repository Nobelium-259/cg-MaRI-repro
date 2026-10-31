"""Fetch a fixed paired diagnostic subset and public distractors, before model evaluation."""
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.parse import quote,urlparse,parse_qs
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor,as_completed
from collections import defaultdict
from io import BytesIO
import argparse,hashlib,json,re,time,zlib,struct,shutil
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/comparison_20261101'
def digest(raw):return hashlib.sha256(raw).hexdigest()
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
def request(url,limit,start=None,end=None):
    headers={'User-Agent':'MaRI-course-reproduction/1.0'}
    if start is not None:headers['Range']=f'bytes={start}-{end}'
    for attempt in range(3):
        try:
            with urlopen(Request(url,headers=headers),timeout=35) as r:
                if start is not None:
                    cr=r.headers.get('Content-Range','')
                    if r.status!=206 or not cr.startswith(f'bytes {start}-'):raise ValueError('Server did not honour bounded range')
                raw=r.read(limit+1);host=urlparse(r.url).hostname
            if len(raw)>limit:raise ValueError('Download cap exceeded')
            return raw,host
        except Exception:
            if attempt==2:raise
            time.sleep(1)
def inspect_image(raw):
    with Image.open(BytesIO(raw)) as im:
        if max(im.size)>8192:raise ValueError('Image dimensions exceed cap')
        im.load();return list(im.size)
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--recipe',type=Path,default=ROOT/'source/course_demo/comparison_recipe.json')
    args=ap.parse_args()
    recipe=json.loads(args.recipe.read_text('utf-8'))
    selection=recipe['selection'];chosen=selection['real_ids'];assets=recipe['ambient_assets']
    groups=selection['real_entries']
    assert len(chosen)==96 and len(assets)==200
    DATA.mkdir(parents=True,exist_ok=True)
    selection_path=DATA/'selection.json'
    if selection_path.exists():
        saved=json.loads(selection_path.read_text('utf-8'))
        if saved!=selection:raise ValueError('Existing selection differs from the frozen recipe')
        selection=saved
    else:write(selection_path,selection)
    write(DATA/'ambient_metadata.json',assets)
    print('FROZEN',digest(selection_path.read_bytes()),len(chosen),len(assets),flush=True)
    archive_url='https://hf-mirror.com/datasets/jianhuiwemi/MaRI/resolve/'+selection['dataset_commit']+'/'+quote(selection['archive']['rfilename'])
    def real_job(folder):
        entries=groups[folder];target=DATA/'real'/folder;target.mkdir(parents=True,exist_ok=True)
        record=target/'source.json'
        if record.exists():
            result=json.loads(record.read_text('utf-8'))
            assert all(digest((target/n).read_bytes())==v['sha256'] for n,v in result['files'].items())
            return result
        # Resume a triplet fetched before QC without redownloading its verified bytes.
        if all((target/n).is_file() and (target/n).stat().st_size==e['size'] and zlib.crc32((target/n).read_bytes())==e['crc32'] for n,e in entries.items()):
            result={'id':folder,'source_archive':selection['archive']['rfilename'],'verification':'ZIP central-directory size/CRC and per-file SHA256; retained bounded-range download','files':{}}
            for n,e in entries.items():
                raw=(target/n).read_bytes();result['files'][n]={'sha256':digest(raw),'size':inspect_image(raw),'bytes':len(raw),'crc32':e['crc32']}
            write(record,result);return result
        start=min(e['offset'] for e in entries.values())
        end=max(e['offset']+e['compressed_size']+65536 for e in entries.values())-1
        assert end-start+1<20*1024**2
        raw,host=request(archive_url,end-start+1,start,end)
        result={'id':folder,'source_archive':selection['archive']['rfilename'],'range_start':start,'range_bytes':len(raw),'resolved_host':host,'files':{}}
        for name,e in entries.items():
            offset=e['offset']-start
            if raw[offset:offset+4]!=b'PK\x03\x04':raise ValueError('ZIP local header mismatch')
            namelen,extra=struct.unpack_from('<HH',raw,offset+26)
            actual_name=raw[offset+30:offset+30+namelen].decode('utf-8')
            if actual_name!=e['name'] or e['method']!=8:raise ValueError('Unexpected ZIP entry')
            pos=offset+30+namelen+extra
            image=zlib.decompress(raw[pos:pos+e['compressed_size']],-15)
            if len(image)!=e['size'] or zlib.crc32(image)!=e['crc32']:raise ValueError('ZIP CRC/size mismatch')
            size=inspect_image(image);(target/name).write_bytes(image)
            result['files'][name]={'sha256':digest(image),'size':size,'bytes':len(image),'crc32':e['crc32']}
        write(record,result);return result
    def asset_job(a):
        aid=a['assetId'];assert re.fullmatch(r'[A-Za-z0-9_]+',aid)
        target=DATA/'ambient'/aid;target.mkdir(parents=True,exist_ok=True)
        record=target/'source.json'
        if record.exists():
            result=json.loads(record.read_text('utf-8'));assert all(digest((target/(n+'.jpg')).read_bytes())==v['sha256'] for n,v in result['files'].items());return result
        query=None
        for link in a.get('previewLinks',[]):
            q=parse_qs(urlparse(link['url']).fragment)
            if q.get('color_url'):query=q['color_url'][0];break
        if not query:raise ValueError('Missing official color preview: '+aid)
        urls={'gallery':a['previewImage']['512-JPG-FFFFFF'],'query':query}
        result={'id':aid,'category':a['displayCategory'],'release_date':a['releaseDate'],'asset_page':a['shortLink'],'license':'CC0-1.0','files':{}}
        for name,url in urls.items():
            raw,host=request(url,5*1024**2);size=inspect_image(raw);(target/(name+'.jpg')).write_bytes(raw)
            result['files'][name]={'url':url,'sha256':digest(raw),'size':size,'bytes':len(raw)}
        write(record,result);return result
    records={'real':{},'ambient':{}};errors=list(recipe.get('frozen_exclusions',[]))
    excluded={e['id'] for e in errors}
    with ThreadPoolExecutor(max_workers=6) as pool:
        tasks={pool.submit(real_job,k):('real',k) for k in chosen}
        tasks.update({pool.submit(asset_job,a):('ambient',a['assetId']) for a in assets if a['assetId'] not in excluded})
        for i,f in enumerate(as_completed(tasks),1):
            kind,key=tasks[f]
            try:records[kind][key]=f.result()
            except Exception as e:errors.append({'kind':kind,'id':key,'error':str(e)})
            if i%10==0 or i==len(tasks):print('ACQUIRE',i,'/',len(tasks),'errors',len(errors),flush=True)
    write(DATA/'acquisition.json',{'selection_sha256':digest(selection_path.read_bytes()),'records':records,'errors':errors})
    fatal=[e for e in errors if e not in recipe.get('frozen_exclusions',[])]
    if fatal:raise RuntimeError('Acquisition incomplete; resume without changing selection: '+str(fatal[:3]))
    # Preserve the exclusion frozen before the original inference, with no substitute.
    if errors:write(DATA/'missing_preview_exclusion.json',{'frozen_before_original_inference':True,'planned_assets':len(assets),'excluded':errors,'replacement':None,'selection_sha256':digest(selection_path.read_bytes())})
    assets=[a for a in assets if a['assetId'] in records['ambient']]
    real_gallery=[{'id':'real_'+k,'path':f'real/{k}/output.png','source':'official_real_pair'} for k in chosen]
    ambient_gallery=[{'id':a['assetId'],'path':f'ambient/{a["assetId"]}/gallery.jpg','category':a['displayCategory'],'source':'ambientCG'} for a in assets]
    queries=[];grouped={};empty_masks=[]
    outputs=defaultdict(list)
    for k in chosen:outputs[records['real'][k]['files']['output.png']['sha256']].append('real_'+k)
    for k in chosen:
        record=records['real'][k]['files'];key=(record['fig.png']['sha256'],record['mask.png']['sha256'])
        with Image.open(DATA/f'real/{k}/mask.png') as im:
            if im.convert('L').getextrema()[1]==0:empty_masks.append(k)
        relevant=outputs[record['output.png']['sha256']]
        if key in grouped:grouped[key]['relevant_ids']=sorted(set(grouped[key]['relevant_ids']+relevant));continue
        q={'id':'real_'+k,'path':f'real/{k}/fig.png','mask':f'real/{k}/mask.png','relevant_ids':list(relevant),'source_group':digest((''.join(key)).encode())}
        grouped[key]=q;queries.append(q)
    different_sizes=[k for k in chosen if records['real'][k]['files']['fig.png']['size']!=records['real'][k]['files']['mask.png']['size']]
    qc={'real_selected':96,'real_unique_queries':len(queries),'ambient_assets':len(assets),'planned_ambient_assets':200,'excluded_before_inference':errors,'empty_masks':empty_masks,'different_resolution_masks':different_sizes,'training_overlap':'unknown; official pairs potentially used in training'}
    write(DATA/'quality_control.json',qc)
    if empty_masks:raise RuntimeError('Empty official mask; needs documented protocol decision')
    common={'warning':'Official public real-world pairs may have trained the released MaRI checkpoint. Diagnostic subset, not held-out generalization or original benchmark. Same full candidate library and preprocessing for all models.','protocol':'official_real96_ambient200_diagnostic_v1','gallery':real_gallery+ambient_gallery,'queries':queries}
    write(DATA/'real_masked.json',common)
    unmasked=json.loads(json.dumps(common));unmasked['protocol']='official_real96_ambient200_unmasked_v1'
    for q in unmasked['queries']:q.pop('mask')
    write(DATA/'real_unmasked.json',unmasked)
    write(DATA/'ambient200.json',{'protocol':'ambient200_flat_to_sphere_v1','warning':'Public CC0 flat-color textures to official sphere previews; controlled representation probe, not real object photographs or official Unseen benchmark. Training overlap unverified.','gallery':ambient_gallery,'queries':[{'id':a['assetId'],'path':f'ambient/{a["assetId"]}/query.jpg','relevant_ids':[a['assetId']]} for a in assets]})
    print('READY',json.dumps(qc),flush=True)
if __name__=='__main__':main()
