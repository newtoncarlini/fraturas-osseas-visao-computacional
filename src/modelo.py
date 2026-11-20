"""Rede multirrótulo e Grad-CAM sobre a saída convolucional do backbone."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
import numpy as np
import tensorflow as tf
from tensorflow import keras

SIZE=160


def create_model(n_classes=10):
    base=keras.applications.MobileNetV2(input_shape=(SIZE,SIZE,3),include_top=False,weights='imagenet')
    base.trainable=False
    head=keras.Sequential([
        keras.layers.Input((1280,)),
        keras.layers.Dense(128,activation='relu',kernel_regularizer=keras.regularizers.l2(.001)),
        keras.layers.Dropout(.45),
        keras.layers.Dense(n_classes,activation='sigmoid')],name='head')
    inputs=keras.Input((SIZE,SIZE,3),name='image_rgb_0_255')
    scaled=keras.layers.Rescaling(1/127.5,offset=-1,name='mobilenet_scale')(inputs)
    features=base(scaled,training=False)
    pooled=keras.layers.GlobalAveragePooling2D(name='pool')(features)
    model=keras.Model(inputs,head(pooled),name='fracture_multilabel')
    return model,base,head


def gradcam(model,image,class_index):
    base=next(layer for layer in model.layers if isinstance(layer,keras.Model) and layer.name!='head')
    head=model.get_layer('head')
    array=tf.convert_to_tensor(image[None,...],dtype=tf.float32)
    # As ativações finais do backbone são os mapas convolucionais.
    with tf.GradientTape() as tape:
        features=base(array/127.5-1,training=False)
        tape.watch(features)
        pooled=tf.reduce_mean(features,axis=(1,2))
        probs=head(pooled,training=False)
        score=probs[:,class_index]
    grads=tape.gradient(score,features)
    weights=tf.reduce_mean(grads,axis=(0,1,2))
    heat=tf.reduce_sum(features[0]*weights,axis=-1)
    heat=tf.maximum(heat,0)
    heat=heat/(tf.reduce_max(heat)+1e-8)
    return heat.numpy()
