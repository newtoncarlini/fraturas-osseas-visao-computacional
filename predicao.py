"""Inferência local. Escores multirrótulo não são probabilidades clínicas calibradas."""
from pathlib import Path
import json, argparse
import numpy as np
import cv2
from PIL import Image,ImageOps
from tensorflow import keras
from src.modelo import gradcam

ROOT=Path(__file__).resolve().parent


def load_artifacts():
    metadata=json.loads((ROOT/'models/metadata.json').read_text(encoding='utf-8'))
    model=keras.models.load_model(ROOT/'models/fraturas.keras',compile=False)
    return model,metadata


def preprocess(image,size):
    rgb=ImageOps.exif_transpose(image).convert('RGB')
    if rgb.width*rgb.height>40_000_000:
        raise ValueError('Imagem muito grande; utilize até 40 milhões de pixels.')
    return cv2.resize(np.asarray(rgb),(size,size),interpolation=cv2.INTER_AREA).astype('float32')


def predict_image(image,model,metadata):
    array=preprocess(image,metadata['image_size'])
    scores=model(array[None,...],training=False).numpy()[0]
    return [{'classe':name,'escore':float(score),'acima_limiar':bool(score>=metadata['threshold'])}
            for name,score in zip(metadata['classes'],scores)],array


def overlay_gradcam(model,array,index):
    heat=gradcam(model,array,index)
    heat=cv2.resize(heat,(array.shape[1],array.shape[0]))
    colored=cv2.cvtColor(cv2.applyColorMap(np.uint8(255*heat),cv2.COLORMAP_JET),cv2.COLOR_BGR2RGB)
    overlay=np.clip(.65*array+.35*colored,0,255).astype('uint8')
    return overlay,heat


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--image',required=True)
    args=parser.parse_args()
    model,metadata=load_artifacts()
    with Image.open(args.image) as im:
        result,_=predict_image(im,model,metadata)
    print(json.dumps(result,ensure_ascii=False,indent=2))
