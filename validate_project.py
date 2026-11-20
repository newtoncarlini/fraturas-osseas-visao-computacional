from pathlib import Path
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
import json,hashlib
import numpy as np
import pandas as pd
import nbformat
from PIL import Image
from predicao import load_artifacts,predict_image,overlay_gradcam
from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score,multilabel_confusion_matrix

ROOT=Path(__file__).resolve().parent


def validate():
    model,meta=load_artifacts(); classes=meta['classes']
    manifest=pd.read_csv(ROOT/'data/manifest.csv'); valid=manifest[manifest.n_labels>0]
    assert valid.groupby('group').split.nunique().max()==1
    assert valid.groupby('pixel_hash').split.nunique().max()==1
    pred=pd.read_csv(ROOT/'reports/previsoes_test.csv')
    truth=pred[[f'true_{c}' for c in classes]].to_numpy()
    scores=pred[[f'prob_{c}' for c in classes]].to_numpy()
    decision=pred[[f'pred_{c}' for c in classes]].to_numpy()
    assert np.isfinite(scores).all() and (scores>=0).all() and (scores<=1).all()
    assert np.array_equal(decision,(scores>=meta['threshold']).astype(int))
    expected=meta['metrics']['test']
    for metric,value in [('f1_macro',f1_score(truth,decision,average='macro',zero_division=0)),
                          ('subset_accuracy',accuracy_score(truth,decision)),
                          ('precision_macro',precision_score(truth,decision,average='macro',zero_division=0)),
                          ('recall_macro',recall_score(truth,decision,average='macro',zero_division=0))]:
        assert np.isclose(value,expected[metric])
    # Recalcula F1 por contagens, de forma independente da rotina de treino.
    tp=(truth*decision).sum(axis=0)
    fp=((1-truth)*decision).sum(axis=0)
    fn=(truth*(1-decision)).sum(axis=0)
    denominator=2*tp+fp+fn
    independent=np.divide(2*tp,denominator,out=np.zeros(len(classes),dtype=float),where=denominator>0).mean()
    assert np.isclose(independent,expected['f1_macro'])
    assert (multilabel_confusion_matrix(truth,decision).sum(axis=(1,2))==len(pred)).all()
    row=pred.iloc[0]
    with Image.open(ROOT/row.path) as image:
        result,array=predict_image(image,model,meta)
    assert np.allclose([r['escore'] for r in result],scores[0],atol=1e-5)
    overlay,heat=overlay_gradcam(model,array,int(np.argmax(scores[0])))
    assert overlay.shape==array.shape and np.isfinite(heat).all() and heat.min()>=0
    assert hashlib.sha256((ROOT/'models/fraturas.keras').read_bytes()).hexdigest()==meta['model_sha256']
    for name in ['01_fraturas_local.ipynb','01_fraturas.ipynb']:
        nb=nbformat.read(ROOT/'notebooks'/name,as_version=4); nbformat.validate(nb)
        code=[c for c in nb.cells if c.cell_type=='code']
        assert all(c.execution_count is not None for c in code)
        assert not any(o.output_type=='error' for c in code for o in c.outputs)
        if name=='01_fraturas.ipynb':
            assert all(not c.outputs for c in code if 'private-images' in c.metadata.get('tags',[]))
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file(str(ROOT/'app.py')).run(timeout=60)
    assert len(app.exception)==0
    status={'status':'aprovado','split_sem_grupos_compartilhados':True,'metricas_recalculadas':True,
            'modelo_recarregado_mesmas_previsoes':True,'gradcam_finito':True,'app_inicial_sem_excecoes':True,
            'notebook_publico_sem_saidas_de_radiografias':True,'teste_imagens':len(pred)}
    (ROOT/'reports/validacao.json').write_text(json.dumps(status,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(status,ensure_ascii=False))


if __name__=='__main__': validate()
