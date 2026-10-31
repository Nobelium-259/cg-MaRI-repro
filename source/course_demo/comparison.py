"""Validated saved comparisons and sequential, CPU-only live queries."""
from pathlib import Path
import gc,json,time
import numpy as np
from core import ROOT,read_manifest,rank,metrics,sha256,make_encoder
from PIL import Image

SPLIT_LABELS={'real_masked':'真实照片＋目标掩码','real_unmasked':'同批真实照片，不加掩码','ambient200':'199项平面纹理→材质球'}
OUTCOME_LABELS={'all':'全部查询','mari_only':'仅 MaRI 首位正确','dinov2_only':'仅 DINOv2 首位正确','both_correct':'两者首位均正确','both_wrong':'两者首位均错误'}
WARNING='官方真实配对可能参与过 MaRI 权重训练，成员关系未核实。这是公开配对子集上的诊断，不能作为独立泛化或论文完整基准的结论。'

def center_view(path,mode='RGB'):
    with Image.open(path) as opened:image=opened.convert(mode)
    w,h=image.size
    size=(256,int(256*h/w)) if w<=h else (int(256*w/h),256)
    image=image.resize(size,Image.Resampling.BILINEAR)
    left=round((image.width-224)/2);top=round((image.height-224)/2)
    return image.crop((left,top,left+224,top+224))

def input_preview(path,mask=None):
    # Denormalize the exact cropped normalized input, after mask multiplication.
    # Normalized zero corresponds to the ImageNet mean, not black pixels.
    image=center_view(path).convert('RGB');pixels=np.asarray(image,dtype=np.float32)/255
    if mask:
        alpha=np.asarray(center_view(mask,'L'),dtype=np.float32)/255
        mean=np.array([.485,.456,.406],dtype=np.float32)
        pixels=(pixels-mean)*alpha[:,:,None]+mean
    return Image.fromarray(np.clip(np.rint(pixels*255),0,255).astype('uint8'))

def local_path(value):
    p=(ROOT/str(value).replace('\\','/')).resolve()
    if not p.is_relative_to(ROOT.resolve()):raise ValueError('Comparison path must stay inside the project')
    return p

def load_comparison(pointer=None):
    pointer=Path(pointer or ROOT/'results/comparison_latest.json')
    ref=json.loads(pointer.read_text('utf-8'))
    folder=local_path(ref['run_directory']);data_dir=local_path(ref['data_directory'])
    if sha256(folder/'summary.json')!=ref['summary_sha256']:raise ValueError('Comparison summary checksum mismatch')
    summary=json.loads((folder/'summary.json').read_text('utf-8'))
    selection=sha256(data_dir/'selection.json')
    if selection!=summary['selection_sha256']:raise ValueError('Frozen selection checksum mismatch')
    models={m:json.loads((folder/(m+'.json')).read_text('utf-8')) for m in ['mari','dinov2']}
    inputs={m:{k.replace('\\','/'):v for k,v in result['inputs_sha256'].items()} for m,result in models.items()}
    if inputs['mari']!=inputs['dinov2']:raise ValueError('Models used different input bytes')
    for relative,expected in inputs['mari'].items():
        if sha256(local_path(relative))!=expected:raise ValueError('An input image or mask changed after inference')
    manifests={s:read_manifest(data_dir/(s+'.json')) for s in SPLIT_LABELS}
    for m,result in models.items():
        if result['selection_sha256']!=selection:raise ValueError('Models used different selections')
        for split,manifest in manifests.items():
            if sha256(data_dir/(split+'.json'))!=result['manifest_sha256'][split]:raise ValueError('Manifest changed after inference')
            ids={x['id'] for x in manifest['gallery']};queries={x['id']:x for x in manifest['queries']}
            records=result['splits'][split]['records']
            if len(records)!=len(queries) or {r['query_id'] for r in records}!=set(queries):raise ValueError('Query sets differ')
            for record in records:
                order=[x['id'] for x in record['ranking']]
                if len(order)!=len(ids) or set(order)!=ids:raise ValueError('Incomplete or duplicate candidate ranking')
                relevant=queries[record['query_id']]['relevant_ids']
                if record['relevant_ids']!=relevant:raise ValueError('Relevance labels differ')
                if record['relevant_rank']!=min(order.index(x)+1 for x in relevant):raise ValueError('Incorrect relevant rank')
            computed=metrics({r['query_id']:[x['id'] for x in r['ranking']] for r in records},manifest['queries'])
            if computed!=result['splits'][split]['metrics'] or computed!=summary['splits'][split]['metrics'][m]:raise ValueError('Displayed metrics disagree with complete rankings')
    for split,manifest in manifests.items():
        mr={r['query_id']:r for r in models['mari']['splits'][split]['records']}
        dr={r['query_id']:r for r in models['dinov2']['splits'][split]['records']}
        cases=summary['splits'][split]['cases']
        if [x['query_id'] for x in cases]!=[x['id'] for x in manifest['queries']]:raise ValueError('Case order or membership changed')
        for case in cases:
            qid=case['query_id'];m,d=mr[qid]['relevant_rank'],dr[qid]['relevant_rank']
            outcome='both_correct' if m==1 and d==1 else 'mari_only' if m==1 else 'dinov2_only' if d==1 else 'both_wrong'
            if (case['mari_rank'],case['dinov2_rank'],case['outcome'])!=(m,d,outcome):raise ValueError('Case outcome disagrees with rankings')
    return {'folder':folder,'data_dir':data_dir,'summary':summary,'models':models,'manifests':manifests}

def records_for(bundle,split,qid):
    return {m:next(r for r in x['splits'][split]['records'] if r['query_id']==qid) for m,x in bundle['models'].items()}

def cases_for(bundle,split,outcome='all'):
    if outcome not in OUTCOME_LABELS:raise ValueError('Unknown outcome filter')
    return [x for x in bundle['summary']['splits'][split]['cases'] if outcome=='all' or x['outcome']==outcome]

def load_bank(bundle,mode,split):
    folder=bundle['folder'];p=folder/(mode+'_features.npz')
    meta=json.loads((folder/(mode+'_features.json')).read_text('utf-8'))
    if sha256(p)!=meta['sha256'] or meta['selection_sha256']!=bundle['summary']['selection_sha256']:raise ValueError('Feature bank checksum or selection mismatch')
    if meta['manifest_sha256']!=bundle['models'][mode]['manifest_sha256']:raise ValueError('Feature bank manifest mismatch')
    with np.load(p,allow_pickle=False) as bank:
        ids=bank['gallery_ids'].tolist();features=bank['gallery'].copy()
    if ids!=meta['gallery_ids'] or len(ids)!=len(set(ids)) or features.ndim!=2 or features.shape[0]!=len(ids) or not np.isfinite(features).all():raise ValueError('Invalid feature bank')
    index={v:i for i,v in enumerate(ids)}
    gallery=bundle['manifests'][split]['gallery']
    hashes={k.replace('\\','/'):v for k,v in bundle['models'][mode]['inputs_sha256'].items()}
    for item in gallery:
        relative=Path(item['path']).relative_to(ROOT).as_posix()
        if sha256(item['path'])!=hashes[relative]:raise ValueError('A gallery image changed after indexing')
    return features[[index[x['id']] for x in gallery]]

def live_compare(bundle,split,path,mask=None,checkpoint=None):
    # One encoder at a time, no CUDA calls, no retained model between requests.
    import torch
    torch.set_num_threads(2)
    answers={};gallery=bundle['manifests'][split]['gallery']
    for mode in ['mari','dinov2']:
        features=load_bank(bundle,mode,split)
        weight=Path(checkpoint or ROOT/'weights/mari.pth') if mode=='mari' else ROOT/'.cache/torch/hub/checkpoints/dinov2_vitl14_pretrain.pth'
        expected=bundle['models'][mode].get('checkpoint_sha256')
        if weight.is_file() and expected and sha256(weight)!=expected:raise ValueError('Live checkpoint differs from the indexed checkpoint')
        enc=None
        try:
            start=time.perf_counter()
            enc=make_encoder(mode,checkpoint or ROOT/'weights/mari.pth',ROOT/'source/dinov2','cpu')
            query=enc.encode(path,'image',mask)
            order,scores=rank(features,query,min(5,len(gallery)))
            answers[mode]={'ranking':[{'id':gallery[i]['id'],'cosine':float(score)} for i,score in zip(order,scores)],'load_and_query_seconds':time.perf_counter()-start}
        finally:
            del enc;gc.collect()
    return answers
