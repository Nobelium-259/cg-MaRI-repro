from pathlib import Path
import tempfile
import streamlit as st
from core import ROOT, read_manifest
from runtime import RetrievalRuntime

st.set_page_config(page_title='MaRI 课程复现', layout='wide')
st.title('MaRI 材质检索演示')
st.caption('实时提取神经网络特征并检索。余弦相似度是特征分数，不是材质成分概率。')
model_labels = {'颜色纹理教学基线（非MaRI）': 'baseline', 'MaRI官方权重': 'mari', 'DINOv2预训练基线': 'dinov2'}
mode = model_labels[st.sidebar.selectbox('模型', list(model_labels))]
manifest = st.sidebar.text_input('数据清单', str(ROOT / 'data/demo_manifest.json'))
checkpoint = st.sidebar.text_input('MaRI权重', str(ROOT / 'weights/mari.pth'))
repo = st.sidebar.text_input('DINOv2源码', str(ROOT / 'source/dinov2'))
device = st.sidebar.selectbox('设备', ['cpu', 'cuda'])


@st.cache_resource
def get_runtime():
    return RetrievalRuntime()


if st.sidebar.button('释放模型和候选缓存'):
    get_runtime().clear()
    st.sidebar.success('缓存已释放，下次检索重新加载。')
try:
    data = read_manifest(manifest)
except Exception as error:
    st.error(str(error))
    st.stop()
st.warning(data.get('warning', '自定义数据不能直接对比论文指标。'))
query_id = st.selectbox('演示查询', [query['id'] for query in data['queries']])
query = next(query for query in data['queries'] if query['id'] == query_id)
upload = st.file_uploader('或上传查询图片', type=['png', 'jpg', 'jpeg'])
if st.button('运行检索', type='primary'):
    temporary = None
    try:
        path, mask = query['path'], query.get('mask')
        if upload:
            with tempfile.NamedTemporaryFile(suffix=Path(upload.name).suffix, delete=False) as stream:
                stream.write(upload.getvalue())
                temporary = stream.name
            path, mask = temporary, None
        with st.spinner('加载模型、建立候选库并提取查询特征；首次运行可能较慢'):
            order, scores = get_runtime().retrieve(mode, checkpoint, repo, device, data['gallery'], path, mask)
        # Streamlit 1.30 can interpret Windows drive letters as URL schemes.
        caption = f'上传查询图（{upload.name}）' if upload else f'内置查询图（{query_id}）'
        st.image(Path(path).read_bytes(), caption=caption, width=250)
        for column, index, score in zip(st.columns(len(order)), order, scores):
            item = data['gallery'][index]
            column.image(Path(item['path']).read_bytes(), caption=f"{item['id']}  {score:.4f}")
        st.write('当前模型：', mode)
        st.write('实际设备：', 'cpu' if mode == 'baseline' else device)
        st.session_state['last_result'] = {'mode': mode, 'query_id': query_id,
            'uploaded': upload is not None, 'device': 'cpu' if mode == 'baseline' else device,
            'ranking': [{'id': data['gallery'][i]['id'], 'cosine': float(score)}
                        for i, score in zip(order, scores)]}
    except Exception as error:
        st.session_state.pop('last_result', None)
        st.error(f'运行失败：{error}。不会自动切换到其他模型。')
    finally:
        if temporary:
            Path(temporary).unlink(missing_ok=True)
