def create_model_2():
    # create model
    seed(2017)
    conv = Sequential()
    conv.add(Conv1D(256, 30, input_shape=(30, 1), activation='relu'))
    conv.add(Conv1D(256, 1, activation='relu'))
    conv.add(Flatten())

    conv.add(Dense(300, activation = 'relu'))
    conv.add(Dense(100, activation = 'relu'))
    conv.add(Dense(2, activation = 'softmax'))

    sgd = SGD(lr = 0.1, momentum = 0.9, decay = 0, nesterov = False)
    
    # Compile model
    conv.compile(loss='categorical_crossentropy', optimizer=sgd)
    return conv