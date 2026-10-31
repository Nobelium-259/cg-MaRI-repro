"""Streamlit execution smoke test; browser upload is checked separately."""
from pathlib import Path
import json
import sys
import argparse
from streamlit.testing.v1 import AppTest
from core import read_manifest

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / 'source/course_demo'))
app = AppTest.from_file(str(root / 'source/course_demo/app.py'), default_timeout=120)
app.run()
if app.exception:
    raise RuntimeError(str(app.exception))
results = []
queries = read_manifest(root / 'data/demo_manifest.json')['queries']
for query in [item['id'] for item in queries]:
    app.selectbox[0].set_value(query).run()
    next(button for button in app.button if button.label == '运行检索').click().run()
    if app.exception or app.error:
        raise RuntimeError(str(app.exception) + str(app.error))
    result = app.session_state['last_result']
    if result['mode'] != 'baseline' or len(result['ranking']) != 5:
        raise RuntimeError('incorrect baseline display')
    results.append(dict(result))
app.sidebar.selectbox[0].set_value('MaRI官方权重').run()
# The real checkpoint may be installed; test a guaranteed absent path explicitly.
import tempfile
missing_test_dir = tempfile.TemporaryDirectory()
next(field for field in app.sidebar.text_input if field.label == 'MaRI权重').set_value(
    str(Path(missing_test_dir.name) / 'missing_mari.pth')).run()
next(button for button in app.button if button.label == '运行检索').click().run()
if app.exception or not app.error or 'MaRI weights required' not in app.error[0].value:
    raise RuntimeError('missing weights must produce an explicit MaRI failure: ' +
                       str([x.value for x in app.error]) + str([x.value for x in app.exception]))
parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path)
args=parser.parse_args()
if args.output:
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'baseline_queries': results, 'missing_mari_error': app.error[0].value,
                           'browser_upload': 'checked separately', 'status': 'pass'}, ensure_ascii=False, indent=2), encoding='utf8')
print(f'PASS: {len(queries)} baseline queries and explicit missing-MaRI failure')
missing_test_dir.cleanup()
