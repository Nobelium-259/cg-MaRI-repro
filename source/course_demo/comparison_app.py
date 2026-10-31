from pathlib import Path
import tempfile
import pandas as pd
import streamlit as st
from core import ROOT
from comparison import load_comparison,cases_for,records_for,live_compare,input_preview,SPLIT_LABELS,OUTCOME_LABELS,WARNING

st.set_page_config(page_title='MaRI 与 DINOv2 对照',layout='wide')
st.title('MaRI 与 DINOv2 材质检索对照')
st.caption('同一张查询、同一候选库，比较任务微调后与通用预训练特征的检索结果。')
try:
    bundle=load_comparison()
except Exception as error:
    st.error(f'对照数据尚未准备好：{error}')
    st.info('先按 README 完成公开数据准备和 CPU 对照评测；原演示可通过 launch_single_demo.bat 打开。')
    st.stop()
st.warning(WARNING)
label=st.selectbox('对照任务',list(SPLIT_LABELS.values()))
split=next(k for k,v in SPLIT_LABELS.items() if v==label)
summary=bundle['summary']['splits'][split];manifest=bundle['manifests'][split]
gallery={x['id']:x for x in manifest['gallery']}
st.caption(f"完整集合：{summary['queries']} 个查询／{summary['gallery_size']} 个候选。下方筛选只改变展示案例，不改变总体统计。")
rows=[]
for mode,title in [('mari','MaRI'),('dinov2','DINOv2'),('baseline','颜色纹理教学基线')]:
    metric=summary['metrics'][mode]
    rows.append({'方法':title,'首位命中':f"{round(metric['hit_at_1']*summary['queries'])}/{summary['queries']} ({metric['hit_at_1']:.1%})",'前5命中':f"{metric['hit_at_5']:.1%}",'mAP':f"{metric['mAP']:.3f}"})
st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
counts=summary['paired_outcomes']
st.write('　｜　'.join(f'{OUTCOME_LABELS[k]}：{v}' for k,v in counts.items()))

def show_query(query):
    cols=st.columns([2,1,2])
    cols[0].image(Path(query['path']).read_bytes(),caption='查询图',width=220)
    caption='模型输入视野：中心裁剪；灰底为掩码外' if query.get('mask') else '模型输入视野：中心裁剪，无掩码'
    cols[1].image(input_preview(query['path'],query.get('mask')),caption=caption,width=190)
    cols[2].image(Path(gallery[query['relevant_ids'][0]]['path']).read_bytes(),caption='数据集指定的对应材质球',width=180)
    if query.get('mask'):
        with st.expander('查看原始目标掩码'):st.image(Path(query['mask']).read_bytes(),width=180)

def show_rankings(records,relevant=()):
    for column,mode,title in zip(st.columns(2),['mari','dinov2'],['MaRI','DINOv2']):
        record=records[mode]
        column.subheader(title+(f"　正确项第 {record['relevant_rank']} 名" if 'relevant_rank' in record else ''))
        for cell,item in zip(column.columns(5),record['ranking'][:5]):
            good=item['id'] in relevant
            cell.image(Path(gallery[item['id']]['path']).read_bytes(),caption=('✓ ' if good else '')+item['id']+f"\n{item['cosine']:.3f}")
    st.caption('绿色勾表示匹配数据集标签。分数为各自特征空间的余弦相似度，不能跨模型比较绝对高低，也不是材质概率。')

saved,live=st.tabs(['已完成实验回放','实时 CPU 对照'])
with saved:
    filter_label=st.selectbox('案例筛选',list(OUTCOME_LABELS.values()))
    outcome=next(k for k,v in OUTCOME_LABELS.items() if v==filter_label)
    cases=cases_for(bundle,split,outcome)
    if cases:
        qid=st.selectbox('查询案例',[x['query_id'] for x in cases],key=split+'_'+outcome)
        query=next(x for x in manifest['queries'] if x['id']==qid)
        show_query(query);records=records_for(bundle,split,qid);show_rankings(records,query['relevant_ids'])
        st.session_state['comparison_display']={'split':split,'query_id':qid,'outcome':outcome,'mari_rank':records['mari']['relevant_rank'],'dinov2_rank':records['dinov2']['relevant_rank']}
    else:st.info('这个集合没有此类案例。照片组中仅 DINOv2 命中的数量为0；可切到纹理控制组查看27个反向获胜案例。')
    st.info('此页回放已保存的实际推理结果，无需加载模型；可先看 MaRI 独自命中的案例，再看 DINOv2 的反例，最后回到完整统计。')
with live:
    st.write('上传一张照片，让两个模型依次在 CPU 上检索当前候选库。首次加载较慢，请等待；完成后释放模型。')
    upload=st.file_uploader('查询照片',type=['jpg','jpeg','png'])
    mask_upload=st.file_uploader('可选目标掩码，与照片保持相同比例',type=['png','jpg','jpeg'])
    if st.button('同时对照两个模型',type='primary',disabled=upload is None):
        st.session_state.pop('live_comparison',None)
        try:
            with tempfile.TemporaryDirectory() as d:
                query=Path(d)/'query.png';query.write_bytes(upload.getvalue())
                mask=None
                if mask_upload:mask=Path(d)/'mask.png';mask.write_bytes(mask_upload.getvalue())
                with st.spinner('依次加载 MaRI、DINOv2 并检索'):
                    answers=live_compare(bundle,split,query,mask)
                st.session_state['live_comparison']={'split':split,'answers':answers,'query_bytes':upload.getvalue()}
        except Exception as error:st.error(f'对照失败：{error}。没有替换为其他模型。')
    answer=st.session_state.get('live_comparison')
    if answer and answer['split']==split:
        st.image(answer['query_bytes'],caption='实时上传查询，无人工标签',width=220)
        show_rankings(answer['answers'])
        st.caption('上传图没有相关性标注，只能定性查看结果，不能计算准确率。')
