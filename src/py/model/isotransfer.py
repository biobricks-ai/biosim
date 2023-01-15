def istransfer():
  i = Input(shape=(dim,), name='i')
  p = Input(shape=(1,), name='pid')
  
  pe = keras.layers.Embedding(input_dim=500, output_dim=natoms, input_length=1,
    name="pid_embedding1",embeddings_initializer=keras.initializers.uniform(0,1))(p)
    
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