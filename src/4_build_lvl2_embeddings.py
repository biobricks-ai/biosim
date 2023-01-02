# use the base embeddings and triplet methods to build property specific embeddings 
import pathlib as pl, sqlite3, math, numpy as np, re, os, sys, importlib
import keras, tensorflow as tf
import pandas as pd
from keras.layers import Input, Dense, Lambda, Dot
from keras.models import Model
from src.py import model as arch

def build_triplets(dataset):
    
    card = dataset.cardinality().numpy()

    d2 = dataset.map(lambda a,p,e,s,arr,v: (tf.strings.to_hash_bucket_fast(s,100e6),v,arr))
    d2 = d2.map(lambda si,v,arr: (si,v,tf.squeeze(arr)))
    d2 = d2.map(lambda si,v,arr: (si,v,arr)).cache()

    def build_trips(si,vi,ai):
        a1 = d2.shuffle(card).filter(lambda sj,vj,aj: tf.not_equal(si,sj) & tf.greater(tf.tensordot(ai,aj,axes=1),0.8))
        pos = a1.filter(lambda sj,vj,aj: tf.equal(vi,vj))
        neg = a1.filter(lambda sj,vj,aj: tf.not_equal(vi,vj))
        return tf.data.Dataset.zip((pos,neg)).map(lambda x,y: (ai,x[2],y[2])).take(100)
        
    trips = d2.interleave(build_trips, cycle_length=8, block_length=100, 
        num_parallel_calls=tf.data.AUTOTUNE,  deterministic=True)

    return trips

# TRAIN EMBEDDINGS =======================================================================
def train_triplet_embedding(trn,val,indim):

    def build_trn_data(basedf,batchsize):

        def build_data(arri,arrj,arrk):
            return ({"i":arri,"j":arrj,"k":arrk,"pid":tf.ones(1)},{"out":tf.ones(1)})

        data = basedf.map(build_data).repeat().batch(batchsize)

        def reshapetrain(x,y):
            xnew = {k:tf.reshape(v,(batchsize,tf.shape(v)[1])) for k,v in x.items()}
            ynew = {k:tf.reshape(v,(batchsize,1)) for k,v in y.items()}
            return (xnew,ynew)

        return data.map(lambda x,y: reshapetrain(x,y))

    batchd = 1000
    ttrips = build_triplets(trn)
    ttrips = build_trn_data(ttrips,batchd).take(1000).cache().prefetch(tf.data.AUTOTUNE)

    it = iter(ttrips)
    x1 = next(it)[0]['i']
    x2 = next(it)[0]['i']

    vtrips = build_trn_data(build_triplets(val),batchd).take(10).cache().prefetch(tf.data.AUTOTUNE)

    tenboard = tf.keras.callbacks.TensorBoard(log_dir='./logs')
    patience = keras.callbacks.EarlyStopping(patience=20,restore_best_weights=True)

    def fit_model():
        importlib.reload(model)
        model = model.build_model(dim)
        model.fit(ttrips,epochs=1000,steps_per_epoch=100,batch_size=batchd, 
            validation_data=vtrips, validation_steps=10, verbose=1, 
            callbacks=[tenboard,patience])
        return model
    
    model = fit_model()
    model.fit(vtrips,epochs=1000,steps_per_epoch=10,batch_size=batchd)

    # WHAT IS THE AVERAGE DISTANCE BETWEEN SIMILAR AND DISSIMILAR EMBEDDINGS==================
    egtrn = next(iter(ttrips))
    egval = next(iter(vtrips))
    
    dtrn = np.mean(model.predict(egtrn[0]))
    dval = np.mean(model.predict(egval[0]))
    print(f"TRAIN: {dtrn} VAL: {dval}")

    # WHAT DO SOME EMBEDDINGS LOOK LIKE?
    w1 = Model(inputs=model.input[3], outputs=model.get_layer('embedding').output)(egtrn[0]['i'])
    w2 = Model(inputs=model.input[3], outputs=model.get_layer('embedding').output)(egval[0]['i'])

    x = next(iter(vtrips))
    np.mean(model.predict(x[0]))

    # STORE THE NEW EMBEDDINGS=================================================================
    pembed = Model(inputs=model.input[3], outputs=model.get_layer('embedding').output) 
    embed = Model(inputs=[model.input[0],model.input[3]], outputs=model.get_layer('embeddingi').output)
    embed([x[0]['i'],x[0]['pid']])
    x = next(iter(hld))
    embed(x[4])

    it = iter(build_trn_data(build_triplets(hld),100))

    x = next(it)
    ij = d([embed(x[0]['i']),embed(x[0]['j'])])
    ij = d([embed(x[0]['i']),embed(x[0]['j'])])
    ik - ij
    np.mean(model.predict(x[0]))
    # stack x[6] and x[7] to get the embedding of the first pair

    a = embed(x[7]) 
    b = embed(x[4]) # O=C(CN1C(=O)c2ccccc2S1(=O)=O)Nc1ccc(S(=O)(=O)N2CCCCCC2)cc1, 0
    while x[5] == 0:
        x = next(it)
    neg = embed(x[4]) # Oc1ccc(NC(=O)c2ccc(N3C(=O)C4CC=C(Cl)CC4C3=O)cc2)cc1, 1

    edf = hld.map(lambda i,pid,emb,smi,arr,val: (i,pid,emb,smi,embed(arr),val))
    pid = next(iter(edf))[1].numpy().decode()

    path = pl.Path(f"cache/chemprop_tfds/embedding={model.name}/pid={pid}")
    os.system(f'rm -r {path}') if path.exists() else None    
    path.mkdir(parents=True, exist_ok=True)
        
    tf.data.Dataset.save(edf, str(path))

# 1. a property embedding could build a relu projection for each property
# 2. the projection can select a set of constrained embeddings for each property
# 3. the projection can then be used to compare property embeddings

def train_transfer_ff(trn,val):
    dim = tf.shape(next(iter(trn))[4]).numpy()[1]
    
    bdim = 32 

    def munge_trn(a,p,e,s,arr,v):
        x = (tf.reshape(arr,(bdim,768)), tf.ones((bdim,1)))
        y = tf.reshape(v,(bdim,1))
        return (x,y)

    btrn = trn.repeat().batch(bdim).map(munge_trn).prefetch(tf.data.AUTOTUNE)
    bval = val.repeat().batch(bdim).map(munge_trn).prefetch(tf.data.AUTOTUNE)
    stps = math.ceil(trn.cardinality().numpy()/bdim)

    x = next(iter(btrn))
    def fit(natoms=3):
        importlib.reload(arch)
        model = arch.transfer_ff(dim,natoms)
        tenboard = tf.keras.callbacks.TensorBoard(log_dir='./logs')
        patience = keras.callbacks.EarlyStopping(patience=5,restore_best_weights=True)
        model.fit(btrn,epochs=1000,steps_per_epoch=stps,batch_size=bdim,
            validation_data=bval, validation_steps=10, verbose=1,
            callbacks=[tenboard,patience])
        return model
    
    model = fit(natoms=10)
    embpid = Model(inputs=model.inputs[1], outputs=model.get_layer('pid_embedding').output)(x[0][1])
    tf.greater(embpid,0.5)
    def embdiff():
        embsub = Model(inputs=model.inputs, outputs=model.get_layer('chem_embedding').output)(x[0])[0]
        inpsub = x[0][0][0]
        print(tf.reduce_sum(tf.square(inpsub - embsub)))
    return emb

def train_ff(trn,val):
    dim = tf.shape(next(iter(trn))[4]).numpy()[1]
    
    bdim = 1000 

    def munge_trn(a,p,e,s,arr,v):
        return (tf.reshape(arr,(bdim,768)),tf.reshape(v,(bdim,1)))

    trnsteps = math.ceil(trn.cardinality().numpy()/bdim)
    btrn = trn.repeat().batch(bdim).map(munge_trn).take(trnsteps).cache().prefetch(tf.data.AUTOTUNE)
    
    valsteps = math.ceil(val.cardinality().numpy()/bdim)
    bval = val.repeat().batch(bdim).map(munge_trn).take(valsteps).cache().prefetch(tf.data.AUTOTUNE)
    

    x = next(iter(btrn))
    def fit():
        importlib.reload(arch)
        model = arch.feedforward(dim)
        tenboard = tf.keras.callbacks.TensorBoard(log_dir='./logs')
        patience = keras.callbacks.EarlyStopping(patience=50,restore_best_weights=True)
        model.fit(btrn,epochs=1000,steps_per_epoch=trnsteps,batch_size=bdim,
            validation_data=bval, validation_steps=valsteps, verbose=1,
            callbacks=[tenboard,patience])
        return model
    
    model = fit()
    

    return Model(inputs=model.inputs, outputs=model.get_layer('embedding').output)


def train():
    
    pat = re.compile('.*/pid=[^/]+$')
    tfpaths = (p.as_posix() for p in pl.Path('cache/chemprop_tfds').glob('**/') if pat.match(p.as_posix()))
    
    path = next(tfpaths)
    pid = path.split('/')[-1].split('=')[-1]
    emb = path.split('/')[-2].split('=')[-1]
    
    src = tf.data.Dataset.load(path)
    src = src.shuffle(src.cardinality(), reshuffle_each_iteration=False)

    chunk = math.ceil(src.cardinality().numpy()*0.1)
    trn = src.take(8*chunk).cache()
    val = src.skip(8*chunk).take(chunk).cache()
    hld = src.skip(9*chunk).cache()

    embed = train_ff(trn,val)
    
    # output should have aid, pid, embedding_type, smiles, array, value
    outpath = pl.Path(f'cache/chemprop_tfds/embedding=isodense/pid={pid}')
    edf = hld.map(lambda i,p,e,smi,arr,v: (i,p,"isodense",smi,tf.squeeze(embed(arr)),v)).cache()
    
    # delete outpath if it exists
    os.system(f'rm -r {outpath}') if outpath.exists() else None    
    tf.data.Dataset.save(edf, str(outpath))

    def build_pairs(si,ai,ei):
        simfn = lambda sj,aj,ej: (si,sj,tf.tensordot(ai,aj,axes=1),tf.tensordot(ei,ej,axes=1))
        return edf.shuffle(edf.cardinality()).map(simfn)

    pairs = edf.interleave(build_pairs)


    it = iter(pairs)
    si, sj, asim, esim = next(it)
    sj, aj, ej = next(it)

    i = 0
    from rdkit import Chem
    
    for si, sj, asim, esim in it:
        dif = asim - esim
        
        # if dif > 0.02 and sim1 > 0.7 and sim1 < 1.0 :s
        smi, smj = (si.numpy(), sj.numpy())
        print(smi)
        print(smj)
        print(dif)
        print()

                # Chem.Draw.MolsToGridImage([Chem.MolFromSmiles(smi),Chem.MolFromSmiles(smj)],
                #     molsPerRow=2,subImgSize=(300,300))

                # mol1 = Chem.MolFromSmiles(si)
                # mol2 = Chem.MolFromSmiles(sj)

                # # Generate an image of the molecule
                # img1 = Chem.Draw.MolToImage(mol1)
                # img2 = Chem.Draw.MolToImage(mol2)