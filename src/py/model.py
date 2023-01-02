import pathlib as pl, sqlite3, math, numpy as np, re, os
import keras, tensorflow as tf
from keras.layers import Input, Dense, Lambda, Dot
from keras.models import Model
import keras.backend as K
import functools


        # diff = soutput-inputs
        # dist = tf.norm(diff,axis=1,keepdims=True) # 0 1. 0.01 should become 0, 0.1, 0.01
        # dist = tf.maximum(1e-8, dist)
        # res =  tf.minimum(dist,self.l2)*diff / dist + inputs
        # self.add_loss(0.0001*tf.reduce_mean(tf.square(inputs-soutput)))
        
class IsoDense(tf.keras.layers.Layer):

    def __init__(self, units, coefficient, activation, **kwargs):
        super(IsoDense, self).__init__(**kwargs)
        self.coefficient = tf.keras.backend.cast_to_floatx(coefficient)
        self.units = units
        self.activation = activation
        self.dense = tf.keras.layers.Dense(self.units, activation=self.activation)

    def build(self, input_shape):
        self.dense.build(input_shape[0])
        
    def call(self, inputs, training=None):
        soutput = self.dense.call(inputs[0])
        soutput = soutput / tf.norm(soutput,axis=1,keepdims=True)
        inpdif = tf.matmul(inputs[1],tf.transpose(inputs[1]))
        outdif = tf.matmul(soutput,tf.transpose(soutput))
        self.add_loss(self.coefficient*tf.norm(inpdif - outdif))
        return soutput
        
    def get_config(self):
        config = {'coefficient': float(self.coefficient)}
        base_config = super(IsoDense, self).get_config()

class DenseCos(tf.keras.layers.Dense):
    
    def __init__(self, l2, **kwargs):
        super(DenseCos, self).__init__(**kwargs)
        self.l2 = tf.keras.backend.cast_to_floatx(l2)

    def call(self, inputs, training=None):
        # TODO this doesn't work. DenseL2 works for now, but DenseCos would be better.
        soutput = super(DenseCos,self).call(inputs)
        soutput = soutput / tf.norm(soutput, axis=1, keepdims=True)
        diff = soutput-inputs
        dist = 1-tf.einsum('ij,ij->i',soutput, inputs)
        return inputs + tf.minimum(tf.reshape(dist,(-1,1)),self.l2)*diff
        
    def get_config(self):
        config = {'l2': float(self.l2)}
        base_config = super(DenseCos, self).get_config()

class ProjLayer(tf.keras.layers.Dense):
    
    def __init__(self, l1, **kwargs):
        super(ProjLayer, self).__init__(**kwargs)
        self.l1 = tf.keras.backend.cast_to_floatx(l1)

    def call(self, inputs, training=None):
        output = super(ProjLayer,self).call(inputs) #+ 0.00001
        outsum = tf.reduce_sum(output,axis=1,keepdims=True)
        return output/outsum
    
    def get_config(self):
        config = {'l1': float(self.l1)}
        base_config = super(ProjLayer, self).get_config()

def transfer_ff(dim,natoms=10):
    i = Input(shape=(dim,), name='i')
    p = Input(shape=(1,), name='pid')
    
    pe = keras.layers.Embedding(input_dim=500, output_dim=natoms, input_length=1,name="pid_embedding1",embeddings_initializer=keras.initializers.uniform(0,1))(p)
    pr = keras.layers.Flatten()(pe)
    pr = ProjLayer(units=natoms, l1=0.01, activation='relu', name='pid_embedding')(pr)
    
    # map each atom to an output layer
    def make_atom(atomi):
        return IsoDense(units=768, activation='tanh', coefficient=0.1,name=f"atom_{atomi}")(i)
    
    atoms = [make_atom(atomi) for atomi in range(natoms)]
    atoms = keras.layers.Concatenate(axis=1)(atoms)
    atoms = keras.layers.Reshape((natoms,768))(atoms)
    
    projection = keras.layers.Dot(axes=1,name="chem_embedding")([pr,atoms])
    dp = keras.layers.Dropout(0.66, name="dropout")(projection)
    out = Dense(1, activation='sigmoid', name='out', kernel_regularizer=keras.regularizers.l2())(dp)
    
    model = Model(inputs=[i,p],outputs=out,name="ff")
    model.compile("adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def feedforward(dim):
    i = Input(shape=(dim,), name='i')

    seq = [
        keras.layers.Dropout(0.33),
        Dense(units=256, activation='relu'),
        keras.layers.Dropout(0.33),
        Dense(units=128, activation='relu'),
    ]
    
    out = functools.reduce(lambda x,y: y(x), seq, i)
    out = IsoDense(units=10, activation='relu', name='embedding', coefficient=0.01)([out,i])
    out = keras.layers.BatchNormalization()(out)
    out = Dense(1, activation='sigmoid', name='out')(out)
    
    model = Model(inputs=[i],outputs=out,name="ff")
    model.compile("adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def less_than_0_frequency(y_true, y_pred):    
    y_pred_binary = K.less(y_pred, 0)
    accuracy = K.mean(K.cast(y_pred_binary, 'float32'))
    return accuracy

def build_triplet_model(dim):
    i = Input(shape=(dim,), name='i')
    j = Input(shape=(dim,), name='j')
    k = Input(shape=(dim,), name='k')
    pid = Input(shape=(1,), name='pid')

    l2  = lambda r: keras.regularizers.l2(r)
    dp1 = keras.layers.Dropout(0.3, name="dropout")
    de1 = Dense(256, activation='relu', name='embedding1')
    de2 = Dense(128, activation='relu', name='embedding')
    de  = lambda l: de2(de1(dp1(l)))

    ei, ej, ek = (de(i),de(j),de(k))

    # K = keras.backend
    # def distance(inputs):
    #     x, y = inputs
    #     return K.sqrt(K.sum(K.square(x - y), axis=-1))

    # ij = tf.keras.layers.Lambda(distance)([ei,ej])
    # ik = tf.keras.layers.Lambda(distance)([ei,ek])

    ij = (1-Dot(axes=1,normalize=True,name="pos")([ei,ej]))
    ik = (1-Dot(axes=1,normalize=True,name="neg")([ei,ek]))

    out = keras.layers.subtract([ij,ik],name="out")

    def triplet_loss(y_true, y_pred):
        margin = 0.05
        return tf.maximum(0., margin + y_pred)

    model = Model(inputs=[i,j,k,pid],outputs=out,name="simple-triplet")
    op = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(op, loss=triplet_loss, metrics=[less_than_0_frequency])
    return model

def siamesemodel(dim):

    i = Input(shape=(dim,), name='i')
    j = Input(shape=(dim,), name='j')

    de3 = Dense(128, activation='tanh', name='embedding3', kernel_regularizer=l2(1e-9))
    de  = lambda l: de3(dp3(de2(dp2(de1(dp1(l))))))

    ei = de(keras.layers.LayerNormalization(center=False,scale=False)(i))
    ej = de(keras.layers.LayerNormalization(center=False,scale=False)(j))
    ek = de(keras.layers.LayerNormalization(center=False,scale=False)(k))

    ij = (1-Dot(axes=1,normalize=True,name="pos")([ei,ej]))
    ik = (1-Dot(axes=1,normalize=True,name="neg")([ei,ek]))

    out = keras.layers.Dense(1, activation='sigmoid', name='out', trainable=False,
        kernel_initializer=keras.initializers.ones(), use_bias=False)(ik-ij)

    model = Model(inputs=[i,j,k,pid],outputs=out,name="simple-triplet")
    optim = keras.optimizers.Adam(learning_rate=1)
    model.compile(optimizer="nadam", loss=keras.losses.binary_crossentropy)
    return model