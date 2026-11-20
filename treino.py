"""Transfer learning: extrai atributos uma vez e treina a cabeça com regularização."""
from pathlib import Path
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS','0')
import json, argparse, hashlib, platform, importlib.metadata
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score,hamming_loss,classification_report,average_precision_score
from src.modelo import create_model,SIZE
from preparar_dados import CLASSES

ROOT=Path(__file__).resolve().parent


def read_rgb(path):
    raw=cv2.imread(str(path),cv2.IMREAD_COLOR)
    if raw is None: raise ValueError(f'Imagem ilegível: {path}')
    return cv2.cvtColor(cv2.resize(raw,(SIZE,SIZE),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)


def train(epochs=60):
    keras.utils.set_random_seed(42)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    manifest=pd.read_csv(ROOT/'data/manifest.csv')
    frame=manifest[manifest.n_labels>0].copy().reset_index(drop=True)
    X=np.stack([read_rgb(ROOT/p) for p in frame.path]).astype('float32')
    Y=frame[CLASSES].to_numpy(dtype='float32')
    ids={s:np.flatnonzero(frame.split==s) for s in ['train','valid','test']}
    model,base,head=create_model(len(CLASSES))
    # GPU não é necessária: backbone congelado; vetores cacheados durante este treino.
    extractor=keras.Model(base.input,keras.layers.GlobalAveragePooling2D()(base.output))
    pooled=extractor.predict(X/127.5-1,batch_size=32,verbose=1)
    rng=np.random.default_rng(42)
    augment=[]
    for image in X[ids['train']]:
        mat=cv2.getRotationMatrix2D((SIZE/2,SIZE/2),float(rng.uniform(-3,3)),1)
        transformed=cv2.warpAffine(image,mat,(SIZE,SIZE),borderMode=cv2.BORDER_REFLECT_101)
        augment.append(np.clip(transformed*rng.uniform(.95,1.05),0,255))
    extra=extractor.predict(np.stack(augment)/127.5-1,batch_size=32,verbose=1)
    features=np.concatenate([pooled[ids['train']],extra])
    labels=np.concatenate([Y[ids['train']],Y[ids['train']]])
    # Cada grupo contribui peso total semelhante, reduzindo domínio de versões aumentadas.
    counts=frame.loc[ids['train']].groupby('group').size()
    sample_weights=frame.loc[ids['train'],'group'].map(1/counts).to_numpy()
    sample_weights=sample_weights/sample_weights.mean()
    sample_weights=np.tile(sample_weights,2)
    positive=labels.sum(axis=0)
    pos_weights=np.clip((len(labels)-positive)/np.maximum(positive,1),1,12).astype('float32')
    def weighted_bce(y_true,y_pred):
        y_pred=tf.clip_by_value(y_pred,1e-7,1-1e-7)
        return tf.reduce_mean(-pos_weights*y_true*tf.math.log(y_pred)-(1-y_true)*tf.math.log(1-y_pred),axis=-1)
    head.compile(optimizer=keras.optimizers.Adam(learning_rate=.0005),loss=weighted_bce,
                 metrics=[keras.metrics.BinaryAccuracy(name='binary_accuracy')])
    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',patience=7,restore_best_weights=True),
               keras.callbacks.ReduceLROnPlateau(monitor='val_loss',patience=3,factor=.5,min_lr=1e-6)]
    history=head.fit(features,labels,sample_weight=sample_weights,validation_data=(pooled[ids['valid']],Y[ids['valid']]),
                     epochs=epochs,batch_size=32,callbacks=callbacks,verbose=2)
    pd.DataFrame(history.history).to_csv(ROOT/'reports/history.csv',index_label='epoch_index')
    prob_valid=head.predict(pooled[ids['valid']],verbose=0)
    # Um limiar global, escolhido SOMENTE na validação. Não há seleção no teste.
    choices=np.arange(.15,.81,.05)
    threshold_scores=[f1_score(Y[ids['valid']],prob_valid>=t,average='macro',zero_division=0) for t in choices]
    threshold=float(choices[int(np.argmax(threshold_scores))])
    pd.DataFrame({'threshold':choices,'valid_f1_macro':threshold_scores}).to_csv(ROOT/'reports/threshold_validation.csv',index=False)
    # Não é preciso salvar a função de perda customizada para inferência.
    # A cabeça aninhada também precisa de uma configuração serializável.
    head.compile(optimizer='adam',loss='binary_crossentropy')
    model.save(ROOT/'models/fraturas.keras')
    reloaded=keras.models.load_model(ROOT/'models/fraturas.keras',compile=False)
    assert np.allclose(reloaded(X[ids['valid'][:3]],training=False).numpy(),prob_valid[:3],atol=1e-5)
    all_metrics={}
    for split in ['train','valid','test']:
        prob=head.predict(pooled[ids[split]],verbose=0)
        truth=Y[ids[split]].astype(int); pred=(prob>=threshold).astype(int)
        baseline=np.zeros(len(CLASSES),dtype=int)
        baseline[int(np.argmax(Y[ids['train']].mean(axis=0)))]=1
        baseline=np.tile(baseline,(len(truth),1))
        report=classification_report(truth,pred,target_names=CLASSES,output_dict=True,zero_division=0)
        pd.DataFrame(report).T.to_csv(ROOT/f'reports/classificacao_{split}.csv')
        all_metrics[split]={'n_images':len(truth),'n_groups':int(frame.iloc[ids[split]].group.nunique()),
             'subset_accuracy':accuracy_score(truth,pred),'f1_macro':f1_score(truth,pred,average='macro',zero_division=0),
             'f1_micro':f1_score(truth,pred,average='micro',zero_division=0),
             'precision_macro':precision_score(truth,pred,average='macro',zero_division=0),
             'recall_macro':recall_score(truth,pred,average='macro',zero_division=0),'hamming_loss':hamming_loss(truth,pred),
             'average_precision_macro':average_precision_score(truth,prob,average='macro'),
             'baseline_f1_macro':f1_score(truth,baseline,average='macro',zero_division=0),
             'baseline_subset_accuracy':accuracy_score(truth,baseline)}
        out=frame.iloc[ids[split]][['path','image','group','split']].reset_index(drop=True)
        for j,c in enumerate(CLASSES):
            out[f'true_{c}']=truth[:,j]; out[f'prob_{c}']=prob[:,j]; out[f'pred_{c}']=pred[:,j]
        out.to_csv(ROOT/f'reports/previsoes_{split}.csv',index=False)
    # Reamostragem por grupos de origem, não por imagens aumentadas.
    test=pd.read_csv(ROOT/'reports/previsoes_test.csv'); groups=test.group.unique()
    boot=[]
    for _ in range(600):
        sampled=rng.choice(groups,size=len(groups),replace=True)
        selected=np.concatenate([np.flatnonzero(test.group==g) for g in sampled])
        yt=test.iloc[selected][[f'true_{c}' for c in CLASSES]].to_numpy()
        yp=test.iloc[selected][[f'pred_{c}' for c in CLASSES]].to_numpy()
        boot.append(f1_score(yt,yp,average='macro',zero_division=0))
    metadata={'architecture':'MobileNetV2 ImageNet congelada + Dense128 L2 + Dropout0.45 + sigmoid10',
      'image_size':SIZE,'input':'RGB float32, 0–255; normalização incorporada ao modelo','classes':CLASSES,
      'task':'multilabel image classification; não é detector de bounding boxes',
      'threshold':threshold,'seed':42,'epochs_executed':len(history.history['loss']),
      'best_epoch_val_loss':int(np.argmin(history.history['val_loss']))+1,'metrics':all_metrics,
      'test_f1_macro_group_bootstrap_95pct':np.quantile(boot,[.025,.975]).tolist(),
      'class_positive_weights':dict(zip(CLASSES,pos_weights.tolist())),
      'limitations':['sem validação clínica','sem IDs de pacientes','grupos inferidos de nomes e pixels','raras classes têm poucos grupos'],
      'model_sha256':hashlib.sha256((ROOT/'models/fraturas.keras').read_bytes()).hexdigest(),
      'python':platform.python_version(),
      'versions':{p:importlib.metadata.version(p) for p in ['tensorflow-cpu','keras','numpy','pandas','scikit-learn','opencv-python-headless']}}
    (ROOT/'models/metadata.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(metadata,ensure_ascii=False,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--epochs',type=int,default=60)
    train(parser.parse_args().epochs)
