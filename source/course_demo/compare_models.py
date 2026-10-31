"""Evaluate fixed MaRI/DINOv2 comparison protocols and retain reusable feature banks."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,re,time,ast,copy,os
import numpy as np
from core import ROOT,read_manifest,make_encoder,rank,metrics,sha256,synchronize

SPLITS=['real_masked','real_unmasked','ambient200']
def write(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8',newline='\n')
def upstream_parity(enc,data):
    """Exercise inspected upstream encode methods, with the same strictly loaded branch weights."""
    from PIL import Image
    from torchvision import transforms
    import torch
    source=ROOT/'source/MaRI_official/inference.py'
    parsed=ast.parse(source.read_text('utf-8'))
    cls=next(x for x in parsed.body if isinstance(x,ast.ClassDef) and x.name=='CLIPLikeMaterialSearchModel')
    namespace={'nn':torch.nn,'torch':torch,'copy':copy}
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(source),'exec'),namespace)
    reference=namespace[cls.name].__new__(namespace[cls.name]);torch.nn.Module.__init__(reference)
    reference.render_encoder=enc.image;reference.material_encoder=enc.material
    image_transform=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    mask_transform=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor()])
    checks=[]
    for branch,items in [('image',data['queries'][:2]),('material',data['gallery'][:4])]:
        for item in items:
            with Image.open(item['path']) as im: x=image_transform(im.convert('RGB')).unsqueeze(0)
            if branch=='image' and item.get('mask'):
                with Image.open(item['mask']) as im:x=x*mask_transform(im.convert('L')).unsqueeze(0)
            with torch.inference_mode():
                fn=reference.encode_render if branch=='image' else reference.encode_material
                expected=fn(x.to(enc.device)).cpu().numpy()[0]
            actual=enc.encode(item['path'],branch,item.get('mask'))
            error=float(np.max(np.abs(expected-actual)))
            if not np.allclose(expected,actual,rtol=1e-5,atol=1e-6):raise ValueError('Upstream feature parity failed')
            checks.append({'branch':branch,'id':item['id'],'max_abs_error':error})
    return {'source_sha256':sha256(source),'checks':checks,'scope':'Independent upstream preprocessing/encode methods using identical already strictly loaded branch weights; not an independent full checkpoint load or full upstream gallery CLI.'}

def evaluate(args):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}',args.run_id):raise ValueError('Unsafe run ID')
    folder=ROOT/'results/local_runs'/args.run_id;folder.mkdir(parents=True,exist_ok=True)
    target=folder/(args.mode+'.json')
    if target.exists():raise FileExistsError('Preserve the existing run: '+str(target))
    data_dir=Path(args.data_dir)
    manifests={k:read_manifest(data_dir/(k+'.json')) for k in SPLITS}
    data=manifests['real_masked'];items=data['gallery']
    result={'mode':args.mode,'created_at_utc':datetime.now(timezone.utc).isoformat(),
            'selection_sha256':sha256(data_dir/'selection.json'),'data_directory':str(data_dir.relative_to(ROOT)),
            'manifest_sha256':{k:sha256(data_dir/(k+'.json')) for k in SPLITS},
            'device':'cpu' if args.mode=='baseline' else args.device,'precision':'FP32','batch_size':1,
            'training':False,'training_overlap':'Official real pairs may occur in released checkpoint training; not held-out generalization.',
            'splits':{},'inputs_sha256':{}}
    for split in manifests.values():
        for item in split['gallery']+split['queries']:
            for field in ['path','mask']:
                if item.get(field):
                    p=Path(item[field]);key=str(p.relative_to(ROOT));result['inputs_sha256'].setdefault(key,sha256(p))
    if args.mode!='baseline':
        import torch
        torch.set_num_threads(args.threads)
    start=time.perf_counter()
    print('LOADING',args.mode,args.device,flush=True)
    enc=make_encoder(args.mode,ROOT/'weights/mari.pth',ROOT/'source/dinov2','cpu' if args.mode=='baseline' else args.device)
    if args.mode=='mari':result['checkpoint_sha256']=enc.checkpoint_sha256
    elif args.mode=='dinov2':result['checkpoint_sha256']=sha256(ROOT/'.cache/torch/hub/checkpoints/dinov2_vitl14_pretrain.pth')
    result['load_seconds']=time.perf_counter()-start
    result['cpu_threads']=args.threads if args.mode!='baseline' else None
    result['pause_seconds_between_items']=args.pause
    print('MODEL_READY',args.mode,flush=True)
    if args.mode=='mari':result['upstream_parity']=upstream_parity(enc,data)
    for _ in range(3):
        enc.encode(items[0]['path'],'material');enc.encode(data['queries'][0]['path'],'image',data['queries'][0].get('mask'))
    synchronize(enc);start=time.perf_counter()
    features=[]
    for n,item in enumerate(items,1):
        features.append(enc.encode(item['path'],'material'))
        time.sleep(args.pause)
        if n%25==0: print('GALLERY',n,'/',len(items),flush=True)
    gallery=np.stack(features);synchronize(enc)
    result['index_seconds']=time.perf_counter()-start
    bank={'gallery':gallery,'gallery_ids':np.array([x['id'] for x in items])}
    index={x['id']:i for i,x in enumerate(items)}
    for split,manifest in manifests.items():
        gitems=manifest['gallery'];g=gallery[[index[x['id']] for x in gitems]]
        records=[];predictions={};features=[]
        for number,q in enumerate(manifest['queries'],1):
            synchronize(enc);start=time.perf_counter();feature=enc.encode(q['path'],'image',q.get('mask'))
            order,scores=rank(g,feature,len(g));synchronize(enc);seconds=time.perf_counter()-start
            time.sleep(args.pause)
            if number%25==0:print('QUERY',split,number,'/',len(manifest['queries']),flush=True)
            ids=[gitems[i]['id'] for i in order];predictions[q['id']]=ids;features.append(feature)
            records.append({'query_id':q['id'],'relevant_ids':q['relevant_ids'],'source_group':q.get('source_group',q['id']),
                            'latency_seconds':seconds,'relevant_rank':min(ids.index(x)+1 for x in q['relevant_ids']),
                            'ranking':[{'id':identifier,'cosine':float(score)} for identifier,score in zip(ids,scores)]})
        result['splits'][split]={'gallery_size':len(gitems),'metrics':metrics(predictions,manifest['queries']),
                                'query_median_seconds':float(np.median([x['latency_seconds'] for x in records])),
                                'records':records}
        bank[split+'_queries']=np.stack(features)
        print(args.mode,split,result['splits'][split]['metrics'],flush=True)
    if hasattr(enc,'torch'):
        result.update({'torch_version':enc.torch.__version__,'cuda_runtime':enc.torch.version.cuda,
                       'gpu_vram_bytes':enc.torch.cuda.get_device_properties(0).total_memory if enc.device.type=='cuda' else None})
    write(target,result)
    bank_path=folder/(args.mode+'_features.npz');np.savez_compressed(bank_path,**bank)
    write(folder/(args.mode+'_features.json'),{'mode':args.mode,'sha256':sha256(bank_path),'selection_sha256':result['selection_sha256'],'gallery_ids':[x['id'] for x in items],'manifest_sha256':result['manifest_sha256']})

def summarize(folder):
    results={m:json.loads((folder/(m+'.json')).read_text('utf-8')) for m in ['baseline','mari','dinov2']}
    if len({x['selection_sha256'] for x in results.values()})!=1:raise ValueError('Data selections differ')
    for field in ['manifest_sha256','inputs_sha256']:
        fingerprints=[json.dumps(x[field],sort_keys=True) for x in results.values()]
        if len(set(fingerprints))!=1:raise ValueError('Models used different '+field)
    summary={'date':'2026-11-01 Asia/Hong_Kong','selection_sha256':results['mari']['selection_sha256'],
             'data_directory':results['mari']['data_directory'],'training':False,
             'timing_note':'CPU two threads per model; DINOv2 and MaRI workers overlapped. Shared desktop CPU, no performance comparison. Gallery indexing includes 50ms item pauses; query timing excludes pauses.',
             'warning':'Official real subset may overlap model training; results characterize this diagnostic subset, not held-out generalization or paper benchmark.',
             'splits':{}}
    for split in SPLITS:
        records={m:{r['query_id']:r for r in v['splits'][split]['records']} for m,v in results.items()}
        ids=list(records['mari']);assert all(set(x)==set(ids) for x in records.values())
        cases=[]
        for qid in ids:
            m,d=records['mari'][qid],records['dinov2'][qid]
            assert m['relevant_ids']==d['relevant_ids']
            mh,dh=m['relevant_rank']==1,d['relevant_rank']==1
            label='both_correct' if mh and dh else 'mari_only' if mh else 'dinov2_only' if dh else 'both_wrong'
            cases.append({'query_id':qid,'outcome':label,'mari_rank':m['relevant_rank'],'dinov2_rank':d['relevant_rank']})
        delta=np.array([int(records['mari'][q]['relevant_rank']==1)-int(records['dinov2'][q]['relevant_rank']==1) for q in ids])
        rng=np.random.default_rng(20261101)
        boot=delta[rng.integers(0,len(delta),size=(10000,len(delta)))].mean(1)
        summary['splits'][split]={'queries':len(ids),'gallery_size':results['mari']['splits'][split]['gallery_size'],
          'metrics':{m:x['splits'][split]['metrics'] for m,x in results.items()},
          'paired_outcomes':{k:sum(c['outcome']==k for c in cases) for k in ['mari_only','dinov2_only','both_correct','both_wrong']},
          'hit1_difference':float(delta.mean()),'descriptive_paired_bootstrap95':np.quantile(boot,[.025,.975]).tolist(),
          'bootstrap_note':'Seed 20261101, 10000 paired query resamples; descriptive only, training overlap and related source images remain limitations.',
          'cases':cases}
    write(folder/'summary.json',summary)
    write(ROOT/'results/comparison_latest.json',{'run_directory':str(folder.relative_to(ROOT)),'data_directory':summary['data_directory'],'summary_sha256':sha256(folder/'summary.json')})
    print(json.dumps({k:{x:v[x] for x in ['queries','gallery_size','metrics','paired_outcomes']} for k,v in summary['splits'].items()},ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['baseline','mari','dinov2']);p.add_argument('--data-dir',default=str(ROOT/'data/comparison_20261101'));p.add_argument('--run-id',default='20261101_comparison_v2_cpu');p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--threads',type=int,default=2);p.add_argument('--pause',type=float,default=0.05);p.add_argument('--summarize',action='store_true');a=p.parse_args()
    if a.summarize:
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}',a.run_id):raise ValueError('Unsafe run ID')
        summarize(ROOT/'results/local_runs'/a.run_id)
    else:
        if not 1<=a.threads<=4 or not 0<=a.pause<=5:p.error('threads must be 1..4; pause must be 0..5 seconds')
        if not a.mode:p.error('--mode is required for inference')
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}',a.run_id):p.error('Unsafe run ID')
        folder=ROOT/'results/local_runs'/a.run_id;folder.mkdir(parents=True,exist_ok=True)
        lock=folder/(a.mode+'.running')
        try:
            with lock.open('x',encoding='utf-8') as stream:stream.write(str(os.getpid()))
        except FileExistsError:
            p.error('This model already has a running worker; preserve it: '+str(lock))
        try:evaluate(a)
        finally:lock.unlink(missing_ok=True)
