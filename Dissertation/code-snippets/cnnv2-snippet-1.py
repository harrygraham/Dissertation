def create_model(batch_size=100, window_height=5):
    
    inputs = Input(shape=(batch_size, 30)) # This returns a tensor
    
    conv1 = Conv1D(32, (window_height), # 32 filters with a window of width 5
    strides=1, 
    padding='causal', # forward in time
    )(inputs) # syntax to chain layers: Layer(...)(PreviousLayer)
    
    fc1 = Dense(64, activation='relu')(conv1)
    predictions = Dense(2, activation='softmax')(fc1)
    
    model = Model(inputs=inputs,
    outputs=predictions)
    model.compile(optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'])
    
    return model