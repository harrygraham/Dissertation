def reshape_to_batches(a, batch_size):
    # pad with zeros if the length is not divisible by the batch_size
    batch_num = np.ceil((float)(a.shape[0]) / batch_size)
    modulo = batch_num * batch_size - a.shape[0]
    if modulo != 0:
        pad = np.zeros((int(modulo), a.shape[1]))
        a = np.vstack((a, pad))
    return np.array(np.split(a, batch_num))