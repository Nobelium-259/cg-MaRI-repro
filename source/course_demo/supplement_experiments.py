"""Deterministic input-condition study and repeated inference timing, without training."""
from pathlib import Path
from datetime import datetime,timezone
from io import BytesIO
import argparse,gc,json,random,time,re
import numpy as np
from PIL import Image,ImageEnhance,ImageFilter
from core import ROOT,read_manifest,make_encoder,rank,metrics,synchronize,sha256

CONDITIONS=['original','brightness_040','brightness_160','contrast_040','blur_4px','jpeg_quality_10','resolution_28px','center_crop_050','padded_gray','padded_gray_mask','padded_color','padded_color_mask']

def transformed(image,condition):
    """Masks identify only inserted padding, not real object segmentation."""
    if condition=='original':return image.copy(),None
    if condition.startswith('brightness_'):return ImageEnhance.Brightness(image).enhance(int(condition.split('_')[1])/100),None
    if condition=='contrast_040':return ImageEnhance.Contrast(image).enhance(.4),None
    if condition=='blur_4px':return image.filter(ImageFilter.GaussianBlur(4)),None
    if condition=='jpeg_quality_10':
        b=BytesIO();image.save(b,format='JPEG',quality=10);b.seek(0)
        with Image.open(b) as im:return im.convert('RGB'),None
    if condition=='resolution_28px':return image.resize((28,28),Image.Resampling.BILINEAR).resize(image.size,Image.Resampling.BILINEAR),None
    if condition=='center_crop_050':
        w,h=image.size;return image.crop((w//4,h//4,w*3//4,h*3//4)),None
    if condition.startswith('padded_'):
        w,h=image.size
        fill=(128,128,128) if 'gray' in condition else (30,140,210)
        out=Image.new('RGB',(w*2,h*2),fill);out.paste(image,(w//2,h//2))
        mask=Image.new('L',out.size,0);mask.paste(255,(w//2,h//2,w//2+w,h//2+h))
        return out,mask if condition.endswith('_mask') else None
    raise ValueError(condition)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--repeats',type=int,default=30)
    p.add_argument('--manifest',type=Path,default=ROOT/'data/demo_manifest.json')
    p.add_argument('--modes',nargs='+',choices=['baseline','mari','dinov2'],default=['baseline','mari','dinov2'])
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=p.parse_args()
    if args.repeats<5:raise ValueError('At least five timing rounds required')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}',args.run_id):raise ValueError('run-id must contain only letters, digits, underscore or hyphen')
    data=read_manifest(args.manifest)
    if any(not q.get('relevant_ids') for q in data['queries']):raise ValueError('All experiment queries require relevant_ids')
    count=len(data['queries'])
    def source_path(path):
        item=Path(path).resolve()
        return str(item.relative_to(ROOT)) if item.is_relative_to(ROOT) else str(item)
    out=ROOT/'results/local_runs'/args.run_id;out.mkdir(parents=True,exist_ok=False)
    images=ROOT/'data/supplementary'/args.run_id;images.mkdir(parents=True,exist_ok=False)
    conditions={};provenance=[]
    for condition in CONDITIONS:
        queries=[]
        for query_index,q in enumerate(data['queries']):
            with Image.open(q['path']) as im:image=im.convert('RGB')
            image,mask=transformed(image,condition)
            name=f'{query_index:04d}__{condition}';path=images/(name+'.png');image.save(path)
            item={'id':q['id'],'path':str(path),'relevant_ids':q['relevant_ids']}
            if mask is not None:
                mp=images/(name+'_mask.png');mask.save(mp);item['mask']=str(mp)
            queries.append(item)
            provenance.append({'query_id':q['id'],'condition':condition,'source':source_path(q['path']),'source_sha256':sha256(q['path']),'derived':str(path.relative_to(ROOT)),'derived_sha256':sha256(path),'mask':str(Path(item['mask']).relative_to(ROOT)) if item.get('mask') else None})
        conditions[condition]=queries
    protocol={'created_at_utc':datetime.now(timezone.utc).isoformat(),'run_id':args.run_id,'conditions':CONDITIONS,'base_queries':count,'gallery_size':len(data['gallery']),'derived_query_conditions':len(CONDITIONS)*count,'independent_new_samples':0,'mask_scope':'Known inserted padding rectangle only; not object segmentation','crop_scope':'Center 50 percent in each dimension; changes field of view','training':False,'seed':20261031,'repeats':args.repeats,'timed_queries_per_model':args.repeats*count,'precision':'FP32','batch_size':1,'resize':256,'center_crop':224,'warmup_per_branch':5,'timing_scope':'IO, preprocessing, CPU-GPU transfer, forward, normalize, full-gallery rank; explicit CUDA synchronization; cached gallery; shared desktop GPU','provenance':provenance,'manifest_sha256':sha256(args.manifest),'source_protocol':data.get('protocol'),'warning':str(data.get('warning',''))+' Deterministic transformations do not increase independent sample count; no conditions selected by results.'}
    (out/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf8')
    all_results={};torch=None
    for mode in dict.fromkeys(args.modes):
        print('START',mode,flush=True)
        start=time.perf_counter()
        enc=make_encoder(mode,ROOT/'weights/mari.pth',ROOT/'source/dinov2','cpu' if mode=='baseline' else args.device)
        synchronize(enc);load=time.perf_counter()-start
        uses_cuda=hasattr(enc,'device') and enc.device.type=='cuda'
        if uses_cuda:
            torch=enc.torch;torch.cuda.reset_peak_memory_stats()
        for _ in range(5):
            enc.encode(data['gallery'][0]['path'],'material');enc.encode(data['queries'][0]['path'],'image')
        synchronize(enc);start=time.perf_counter()
        gallery=np.stack([enc.encode(x['path'],'material') for x in data['gallery']]);synchronize(enc)
        index=time.perf_counter()-start
        rec={}
        for condition,queries in conditions.items():
            preds={};records=[]
            for q in queries:
                feature=enc.encode(q['path'],'image',q.get('mask'));order,scores=rank(gallery,feature,len(gallery))
                preds[q['id']]=[data['gallery'][i]['id'] for i in order]
                records.append({'query_id':q['id'],'ranking':[{'id':data['gallery'][i]['id'],'cosine':float(s)} for i,s in zip(order,scores)],'relevant_rank':min(preds[q['id']].index(i)+1 for i in q['relevant_ids'])})
            rec[condition]={'metrics':metrics(preds,queries),'records':records}
        latencies=[];schedule=[];rng=random.Random(20261031)
        for repeat in range(args.repeats):
            qs=list(data['queries']);rng.shuffle(qs)
            for q in qs:
                synchronize(enc);start=time.perf_counter()
                feature=enc.encode(q['path'],'image',q.get('mask'));rank(gallery,feature,len(gallery));synchronize(enc)
                elapsed=time.perf_counter()-start;latencies.append(elapsed)
                schedule.append({'round':repeat+1,'query_id':q['id'],'seconds':elapsed})
        memory=None
        if uses_cuda:
            memory={'capacity_bytes':torch.cuda.get_device_properties(0).total_memory,'allocated_bytes':torch.cuda.memory_allocated(),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'scope':'This process PyTorch tensor allocations, not full graphics-device memory use'}
        result={'mode':mode,'device':'cpu' if mode=='baseline' else args.device,'numeric_precision':'FP32','batch_size':1,'model_load_seconds_including_checkpoint_hash':load,'gallery_build_seconds':index,'conditions':rec,'timing':{'n':len(latencies),'rounds':args.repeats,'median_ms':float(np.median(latencies)*1000),'p95_ms':float(np.percentile(latencies,95)*1000),'min_ms':float(min(latencies)*1000),'max_ms':float(max(latencies)*1000),'records':schedule},'memory':memory,'checkpoint_sha256':getattr(enc,'checkpoint_sha256',None)}
        (out/(mode+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
        all_results[mode]=result
        # Move outgoing parameters off CUDA before deleting Python references.
        # This keeps subsequent per-model allocation measurements isolated.
        if hasattr(enc,'torch'):
            enc.image.to('cpu')
            if enc.material is not enc.image:enc.material.to('cpu')
        del enc;gc.collect()
        if uses_cuda:
            torch.cuda.empty_cache()
            if torch.cuda.memory_allocated()>50*1024**2:
                raise RuntimeError('Outgoing model retained CUDA allocation')
        print('DONE',mode,'median_ms',result['timing']['median_ms'],flush=True)
    summary={'protocol':{k:v for k,v in protocol.items() if k!='provenance'},'models':{m:{'conditions':{c:r['metrics'] for c,r in d['conditions'].items()},'timing':{k:v for k,v in d['timing'].items() if k!='records'},'memory':d['memory']} for m,d in all_results.items()}}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
