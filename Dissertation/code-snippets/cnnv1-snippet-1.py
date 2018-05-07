# Function to create model
def create_model():
    # create model
    seed(2017)
    conv = Sequential()
    conv.add(Conv1D(256, 30, input_shape=(30, 1), activation='relu'))
    conv.add(Flatten())
    conv.add(Dense(300, activation = 'relu'))
    conv.add(Dense(2, activation = 'softmax'))

    adam = Adam(lr=0.1)
    
    # Compile model
    conv.compile(loss='categorical_crossentropy', optimizer=sgd)
    return conv