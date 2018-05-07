def generate_train_test_timeseries(data, test_ratio=0.3):
    '''data: Python Pandas DataFrame containing vector data and their class sort_values.
       test_ratio: The ratio of train:test split.
    '''
    # Get the total number of transactions
    total_samples = data.shape[0]
    # Determine a cutoff point based on the given test_ratio
    cutoff = int(total_samples * (1 - test_ratio))

    # Sort the data based on the Time column
    data.sort_values('Time', inplace=True)
    
    # Split the DataFrame data based on the cutoff point, to preserve time order. 
    # 0 : Cutoff = Train.  Cutoff : End = Test.
    X_train = data.loc[0:cutoff, data.columns != 'Class']
    y_train = data.loc[0:cutoff, data.columns == 'Class']
    X_test = data.loc[cutoff:, data.columns != 'Class']
    y_test = data.loc[cutoff:, data.columns == 'Class']
    
    # Use pipeline and scaler to scale (normalise) the train and test data 
    pipeline = Pipeline([('scaling', StandardScaler())])
    preprocessor = pipeline.fit(X_train)
    X_train_prp = preprocessor.transform(X_train)
    X_test_prp = preprocessor.transform(X_test)
    
    # Return the appropriately ordered and normalised train and test data
    return X_train_prp, y_train, X_test_prp, y_test