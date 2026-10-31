from pathlib import Path
import json,sys
from streamlit.testing.v1 import AppTest
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'source/course_demo'))
from comparison import load_comparison,SPLIT_LABELS,OUTCOME_LABELS
b=load_comparison();app=AppTest.from_file(str(R/'source/course_demo/comparison_app.py'),default_timeout=90)
app.run();checks=[]
if app.exception or app.error:raise RuntimeError(str(app.exception)+str(app.error))
for split,label in SPLIT_LABELS.items():
    app.selectbox[0].set_value(label).run()
    for outcome,title in OUTCOME_LABELS.items():
        app.selectbox[1].set_value(title).run()
        if app.exception or app.error:raise RuntimeError(str(app.exception)+str(app.error))
        cases=[c for c in b['summary']['splits'][split]['cases'] if outcome=='all' or c['outcome']==outcome]
        if cases:
            for case in [cases[0],cases[-1]]:
                app.selectbox[2].set_value(case['query_id']).run()
                actual=app.session_state['comparison_display']
                assert actual=={'split':split,'query_id':case['query_id'],'outcome':outcome,'mari_rank':case['mari_rank'],'dinov2_rank':case['dinov2_rank']},actual
                checks.append(actual)
        elif not app.info:raise RuntimeError('No explicit empty-case message')
out=b['folder']/'app_check.json'
out.write_text(json.dumps({'status':'pass','checks':checks,'gpu_inference':False,'live_upload':'checked separately in a real browser'},ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
print('APP_PASS',len(checks),'representative first/last cases across 15 groups')
