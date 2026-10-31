from pathlib import Path
from urllib.request import urlopen,Request
from urllib.parse import urlencode
from html.parser import HTMLParser
import hashlib
ROOT=Path(__file__).resolve().parents[2]
SHA='6d7bf5bee85ab3e113ad729f7dfe374285e818693da24e8276c72dd0bd64fb0f'
class Form(HTMLParser):
 def __init__(self):super().__init__();self.action=None;self.data={}
 def handle_starttag(self,tag,attrs):
  a=dict(attrs)
  if tag=='form':self.action=a.get('action')
  if tag=='input' and 'name' in a:self.data[a['name']]=a.get('value','')
def main():
 target=ROOT/'weights/mari.pth';target.parent.mkdir(exist_ok=True)
 if target.exists():raise SystemExit('Target exists. Verify or rename it first.')
 r=urlopen(Request('https://drive.google.com/uc?export=download&id=1rlHUrPDShA_w1OmJF8Eeo-SMX_-CJaGD',headers={'User-Agent':'Mozilla/5.0'}),timeout=90)
 if 'text/html' in r.headers.get('Content-Type',''):
  f=Form();f.feed(r.read().decode());r.close()
  if not f.action or not f.action.startswith('https://drive.usercontent.google.com/'):raise RuntimeError('Use the official README download link manually.')
  r=urlopen(f.action+'?'+urlencode(f.data),timeout=90)
 if 'text/html' in r.headers.get('Content-Type',''):raise RuntimeError('HTML received instead of weights')
 h=hashlib.sha256();partial=target.with_suffix('.partial');total=0
 with r,open(partial,'wb') as out:
  while True:
   chunk=r.read(8*1024*1024)
   if not chunk:break
   h.update(chunk);out.write(chunk);total+=len(chunk)
   if total%(256*1024*1024)==0:print(round(total/1024**3,2),'GiB',flush=True)
 if h.hexdigest()!=SHA:raise RuntimeError('Checksum mismatch; do not load the partial file')
 partial.rename(target);print('Verified:',target)
if __name__=='__main__':main()
