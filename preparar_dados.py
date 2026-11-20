"""Auditoria YOLO e divisão por grupos de origem inferidos, sem misturar aumentos."""
from pathlib import Path
import json, hashlib, zipfile, collections
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit

ROOT=Path(__file__).resolve().parent
CLASSES=['Comminuted','Greenstick','Healthy','Linear','Oblique Displaced','Oblique','Segmental','Spiral','Transverse Displaced','Transverse']
SOURCE='https://cdn3.gnarususercontent.com.br/5327-checkpoint/Human%20Bone%20Fractures%20Multi-modal%20Image%20Dataset%20%28HBFMID%29.zip'


def prepare():
    archive=ROOT/'data/dataset.zip'
    if not archive.exists():
        from urllib.request import urlretrieve
        archive.parent.mkdir(parents=True,exist_ok=True)
        urlretrieve(SOURCE,archive)
    extracted=ROOT/'data/raw'
    if not list(extracted.rglob('*.jpg')):
        with zipfile.ZipFile(archive) as z:
            for item in z.infolist():
                destination=(extracted/item.filename).resolve()
                if not destination.is_relative_to(extracted.resolve()):
                    raise ValueError('Caminho inválido no ZIP')
            z.extractall(extracted)
    rows=[]; annotations=[]; issues=[]
    for image in sorted(extracted.rglob('*.jpg')):
        label=image.parent.parent/'labels'/f'{image.stem}.txt'
        with Image.open(image) as im:
            im.load(); w,h=im.size
            pixels=np.asarray(im.convert('RGB'))
        labels=set()
        for line in label.read_text().splitlines() if label.exists() else []:
            if not line.strip(): continue
            values=line.split()
            if len(values)!=5:
                issues.append({'image':image.name,'issue':'linha YOLO com formato inesperado'}); continue
            c=int(values[0]); xc,yc,bw,bh=map(float,values[1:])
            if not (0<=c<len(CLASSES) and 0<=xc<=1 and 0<=yc<=1 and 0<bw<=1 and 0<bh<=1):
                issues.append({'image':image.name,'issue':'anotação fora dos limites'}); continue
            labels.add(c)
            annotations.append({'image':image.name,'class_id':c,'x':xc,'y':yc,'w':bw,'h':bh})
        row={'path':image.relative_to(ROOT).as_posix(),'image':image.name,'original_split':image.parent.parent.name,
             'source_name':image.stem.split('.rf.')[0], 'width':w,'height':h,'n_labels':len(labels),
             'pixel_hash':hashlib.sha256(pixels.tobytes()).hexdigest()}
        row.update({c:int(i in labels) for i,c in enumerate(CLASSES)})
        rows.append(row)
    frame=pd.DataFrame(rows)
    # União transitiva: nomes de origem compartilhados OU pixels idênticos ficam juntos.
    parent=list(range(len(frame)))
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def union(a,b): parent[find(a)]=find(b)
    for column in ['source_name','pixel_hash']:
        for _,group in frame.groupby(column):
            ids=group.index.tolist()
            for i in ids[1:]: union(ids[0],i)
    frame['group']=[f'g{find(i):04d}' for i in frame.index]
    cross=frame.groupby('group').original_split.nunique().gt(1).sum()
    frame['split']='excluida_sem_rotulo'
    valid=frame[frame.n_labels>0].copy()
    Y=valid[CLASSES].to_numpy()
    best=None
    # Busca determinística apenas de equilíbrio de rótulos, sem resultados de modelos.
    outer=GroupShuffleSplit(n_splits=180,test_size=.2,random_state=42)
    for trainval,test in outer.split(valid,groups=valid.group):
        inside=GroupShuffleSplit(n_splits=8,test_size=.25,random_state=17)
        for tr,va in inside.split(valid.iloc[trainval],groups=valid.iloc[trainval].group):
            train,val=trainval[tr],trainval[va]
            parts=[train,val,test]
            counts=np.array([Y[p].sum(axis=0) for p in parts])
            if (counts<1).any(): continue
            fractions=np.array([.6,.2,.2])[:,None]
            score=np.abs(counts/np.maximum(Y.sum(axis=0),1)-fractions).mean()
            score+=sum(abs(len(p)/len(valid)-f) for p,f in zip(parts,[.6,.2,.2]))
            if best is None or score<best[0]: best=(score,parts)
    if best is None: raise ValueError('Não foi possível obter todas as classes nas três partições.')
    for name,part in zip(['train','valid','test'],best[1]):
        frame.loc[valid.iloc[part].index,'split']=name
    used=frame[frame.n_labels>0]
    assert used.groupby('group').split.nunique().max()==1
    assert used.groupby('pixel_hash').split.nunique().max()==1
    (ROOT/'reports').mkdir(exist_ok=True)
    frame.to_csv(ROOT/'data/manifest.csv',index=False)
    pd.DataFrame(annotations).to_csv(ROOT/'data/annotations.csv',index=False)
    pd.DataFrame(issues,columns=['image','issue']).to_csv(ROOT/'reports/annotation_issues.csv',index=False)
    distribution=used.groupby('split')[CLASSES].sum().reindex(['train','valid','test'])
    distribution.to_csv(ROOT/'reports/distribuicao_classes.csv')
    audit={'images':len(frame),'used_images':len(used),'empty_images_excluded':int((frame.n_labels==0).sum()),
           'multilabel_images':int((frame.n_labels>1).sum()),'groups':int(used.group.nunique()),
           'groups_crossing_original_splits':int(cross),'exact_duplicate_excess':int(frame.pixel_hash.duplicated().sum()),
           'annotation_issues':len(issues),'split_sizes':used.split.value_counts().to_dict(),
           'source':SOURCE,'zip_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
           'group_rule':'prefixo antes de .rf. OU hash de pixels idênticos, união transitiva',
           'limitation':'Não há IDs de pacientes. Nomes podem colidir; agrupamento conservador não garante independência entre pacientes.',
           'license_in_archive':'Private; imagens originais e grids mantidos fora do versionamento'}
    (ROOT/'reports/auditoria.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False,indent=2)); print(distribution.to_string())
    return frame

if __name__=='__main__': prepare()
