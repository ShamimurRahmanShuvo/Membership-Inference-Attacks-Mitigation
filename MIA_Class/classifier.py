from sklearn.metrics import classification_report, accuracy_score
import theano.tensor as T
import numpy as np
import lasagne
import theano
import argparse
import pandas as pd


def iterate_minibatches(inputs, targets, batch_size, shuffle=True):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        #print("Indices", indices)
        np.random.shuffle(indices)

    start_idx = None
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]

    if start_idx is not None and start_idx + batch_size < len(inputs): 
        excerpt = indices[start_idx + batch_size:] if shuffle else slice(start_idx + batch_size, len(inputs))
        yield inputs[excerpt], targets[excerpt]


def get_nn_model(n_in,n_out,n_hidden):
    net = dict()
    net['input'] = lasagne.layers.InputLayer(shape=(None, n_in))
    net['fc'] = lasagne.layers.DenseLayer(
        net['input'],
        num_units=n_hidden,
        nonlinearity=lasagne.nonlinearities.tanh)
    #net['fc1'] = lasagne.layers.DenseLayer(
        #net['fc'],
        #num_units=50,
        #nonlinearity=lasagne.nonlinearities.tanh)
    net['output'] = lasagne.layers.DenseLayer(
        net['fc'],
        num_units=n_out,
        nonlinearity=lasagne.nonlinearities.softmax)
    return net

def get_nn_model1(n_in,n_out,n_hidden):
    net = dict()
    net['input'] = lasagne.layers.InputLayer(shape=(None, n_in))
    #Noise in the input layer
    net['fc'] = lasagne.layers.GaussianNoiseLayer(net['input'], sigma=0)
    net['fc1'] = lasagne.layers.DenseLayer(
        net['fc'],
        num_units=n_hidden,
        nonlinearity=lasagne.nonlinearities.tanh)
    net['fc2'] = lasagne.layers.DenseLayer(
        net['fc1'],
        num_units=50,
        nonlinearity=lasagne.nonlinearities.tanh)
    # Noise in Hidden layer
    #net['fc2'] = lasagne.layers.GaussianNoiseLayer(net['fc1'], sigma=0) 
    net['output'] = lasagne.layers.DenseLayer(
        net['fc2'],
        num_units=n_out,
        nonlinearity=lasagne.nonlinearities.softmax)
    return net

DATA_PATH = './data/'
def get_softmax_model(n_in, n_out):
    net = dict()
    net['input'] = lasagne.layers.InputLayer((None, n_in))
    net['output'] = lasagne.layers.DenseLayer(
        net['input'],
        num_units=n_out,
        nonlinearity=lasagne.nonlinearities.softmax)
    return net
	
#def load_data(data_name):
    #with np.load(DATA_PATH + data_name) as f:
       # train_x, train_y, test_x, test_y = [f['arr_%d' % i] for i in range(len(f.files))]
    #return train_x, train_y, test_x, test_y

def train(dataset, n_hidden=50, batch_size=100, epochs=100, learning_rate=0.01, model='nn', l2_ratio=1e-7,
          rtn_layer=True, noise=True):
		  
    train_x, train_y, test_x, test_y =dataset
    min_y = np.amin(train_y)
    train_y =train_y - min_y
#    print(train_x.shape)
#    print(test_x.shape)
#    print(np.amax(train_x))
#    print(np.amax(test_x))
    
    
    print(train_x.shape)
    print(test_x.shape)
    n_in = train_x.shape[1]
    n_out = len(np.unique(train_y))

    if batch_size > len(train_y):
        batch_size = len(train_y)

    print ('Building model with {} training data, {} classes...'.format(len(train_x), n_out))
    input_var = T.matrix('x')
    target_var = T.ivector('y')
    
    if model == 'nn':
        print ('Using neural network...')
        net = get_nn_model(n_in,n_out,n_hidden)
    else:
        print ('Using softmax regression...')
        net = get_softmax_model(n_in, n_out)

    net['input'].input_var = input_var
    output_layer = net['output']

    #if noise == True:
     #   output_layer = lasagne.layers.GaussianNoiseLayer(output_layer, sigma=1.5)
    #else:
     #  output_layer = output_layer

    # create loss function
    prediction = lasagne.layers.get_output(output_layer)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean() + l2_ratio * lasagne.regularization.regularize_network_params(output_layer,
                                                                                 lasagne.regularization.l2)
    # create parameter update expressions
    params = lasagne.layers.get_all_params(output_layer, trainable=True)
    updates = lasagne.updates.adam(loss, params, learning_rate=learning_rate)
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    # use trained network for predictions
    test_prediction = lasagne.layers.get_output(output_layer, deterministic=True)
    test_fn = theano.function([input_var], test_prediction)

    
    print ('Training ...')
    for epoch in range(epochs):
        loss = 0
        for input_batch, target_batch in iterate_minibatches(train_x, train_y, batch_size):
            loss += train_fn(input_batch, target_batch)
        loss = round(loss, 3)
        #print ('Epoch {}, train loss {}'.format(epoch, loss))

    pred_y = []
    for input_batch, _ in iterate_minibatches(train_x, train_y, batch_size, shuffle=False):
        pred = test_fn(input_batch)
        pred_y.append(np.argmax(pred, axis=1))
    pred_y = np.concatenate(pred_y)

    print ('Training Accuracy: {}'.format(accuracy_score(train_y, pred_y)))
    print (classification_report(train_y, pred_y))

    if test_x is not None:
        print ('Testing...')
        pred_y = []

        if batch_size > len(test_y):
            batch_size = len(test_y)

        for input_batch, _ in iterate_minibatches(test_x, test_y, batch_size, shuffle=False):
            pred = test_fn(input_batch)
            pred_y.append(np.argmax(pred, axis=1))
        pred_y = np.concatenate(pred_y)
        print ('Testing Accuracy: {}'.format(accuracy_score(test_y, pred_y)))
        print (classification_report(test_y, pred_y))

    # return the query function
    if rtn_layer:
        return output_layer
    else:
        return pred_y

def train2(dataset, n_hidden=50, batch_size=100, epochs=100, learning_rate=0.01, model='nn', l2_ratio=1e-7,
          rtn_layer=True, noise=True):
		  
    train_x, train_y, test_x, test_y =dataset
    min_y = np.amin(train_y)
    train_y =train_y - min_y
    train_x=train_x.reshape(-1,1)
    test_x=test_x.reshape(-1,1)
    print(train_x.shape)
    print(np.amax(test_x))
    
    n_in = train_x.shape[1]
#    print('n_in:',n_in)
    n_out = len(np.unique(train_y))

    if batch_size > len(train_y):
        batch_size = len(train_y)

    print ('Building model with {} training data, {} classes...'.format(len(train_x), n_out))
    input_var = T.matrix('x')
    target_var = T.ivector('y')
    
    if model == 'nn':
        print ('Using neural network...')
        net = get_nn_model(n_in,n_out,n_hidden)
    else:
        print ('Using softmax regression...')
        net = get_softmax_model(n_in, n_out)

    net['input'].input_var = input_var
    output_layer = net['output']

    #if noise == True:
     #   output_layer = lasagne.layers.GaussianNoiseLayer(output_layer, sigma=1.5)
    #else:
     #  output_layer = output_layer

    # create loss function
    prediction = lasagne.layers.get_output(output_layer)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean() + l2_ratio * lasagne.regularization.regularize_network_params(output_layer,
                                                                                 lasagne.regularization.l2)
    # create parameter update expressions
    params = lasagne.layers.get_all_params(output_layer, trainable=True)
    updates = lasagne.updates.adam(loss, params, learning_rate=learning_rate)
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    # use trained network for predictions
    test_prediction = lasagne.layers.get_output(output_layer, deterministic=True)
    test_fn = theano.function([input_var], test_prediction)

    
    print ('Training ...')
    for epoch in range(epochs):
        loss = 0
        for input_batch, target_batch in iterate_minibatches(train_x, train_y, batch_size):
            loss += train_fn(input_batch, target_batch)
        loss = round(loss, 3)
        #print ('Epoch {}, train loss {}'.format(epoch, loss))

    pred_y = []
    for input_batch, _ in iterate_minibatches(train_x, train_y, batch_size, shuffle=False):
        pred = test_fn(input_batch)
        pred_y.append(np.argmax(pred, axis=1))
    pred_y = np.concatenate(pred_y)

    print ('Training Accuracy: {}'.format(accuracy_score(train_y, pred_y)))
    print (classification_report(train_y, pred_y))

    if test_x is not None:
        print ('Testing...')
        pred_y = []

        if batch_size > len(test_y):
            batch_size = len(test_y)

        for input_batch, _ in iterate_minibatches(test_x, test_y, batch_size, shuffle=False):
            pred = test_fn(input_batch)
            pred_y.append(np.argmax(pred, axis=1))
        pred_y = np.concatenate(pred_y)
        print ('Testing Accuracy: {}'.format(accuracy_score(test_y, pred_y)))
        print (classification_report(test_y, pred_y))

    # return the query function
    if rtn_layer:
        return output_layer
    else:
        return pred_y
    
    
def train1(dataset, n_hidden=50, batch_size=100, epochs=100, learning_rate=0.01, model='nn', l2_ratio=1e-7,
          rtn_layer=True, noise=True):  
    train_x, train_y, test_x, test_y =dataset
    min_y = np.amin(train_y)
    train_y =train_y - min_y
    print("For Target Model")
    print(train_x.shape)
    print(test_x.shape)
    train_x=train_x.reshape(-1,1)
    test_x=test_x.reshape(-1,1)
    n_in = train_x.shape[1]
    n_out = len(np.unique(train_y))

    if batch_size > len(train_y):
        batch_size = len(train_y)

    print ('Building model with {} training data, {} classes...'.format(len(train_x), n_out))
    input_var = T.matrix('x')
    target_var = T.ivector('y')
    
    if model == 'nn':
        print ('Using neural network...')
        net = get_nn_model1(n_in,n_out,n_hidden)
    else:
        print ('Using softmax regression...')
        net = get_softmax_model(n_in, n_out)

    net['input'].input_var = input_var
    output_layer = net['output']
    
######
#######         Noise in Output Layer
######
    #if noise == True:
       # output_layer = lasagne.layers.GaussianNoiseLayer(output_layer, sigma=0)
    #else:
       #output_layer = output_layer

    # create loss function
    prediction = lasagne.layers.get_output(output_layer)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean() + l2_ratio * lasagne.regularization.regularize_network_params(output_layer,
                                                                                 lasagne.regularization.l2)
    # create parameter update expressions
    params = lasagne.layers.get_all_params(output_layer, trainable=True)
    updates = lasagne.updates.adam(loss, params, learning_rate=learning_rate)
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    # use trained network for predictions
    test_prediction = lasagne.layers.get_output(output_layer, deterministic=True)
    test_fn = theano.function([input_var], test_prediction)

    
    print ('Training Target Model...')
    for epoch in range(epochs):
        loss = 0
        for input_batch, target_batch in iterate_minibatches(train_x, train_y, batch_size):
            loss += train_fn(input_batch, target_batch)
        loss = round(loss, 3)
        #print ('Epoch {}, train loss {}'.format(epoch, loss))

    pred_y = []
    for input_batch, _ in iterate_minibatches(train_x, train_y, batch_size, shuffle=False):
        pred = test_fn(input_batch)
        pred_y.append(np.argmax(pred, axis=1))
    pred_y = np.concatenate(pred_y)

    print ('Training Accuracy: {}'.format(accuracy_score(train_y, pred_y)))
    print (classification_report(train_y, pred_y))

    if test_x is not None:
        print ('Testing...')
        pred_y = []

        if batch_size > len(test_y):
            batch_size = len(test_y)

        for input_batch, _ in iterate_minibatches(test_x, test_y, batch_size, shuffle=False):
            pred = test_fn(input_batch)
            pred_y.append(np.argmax(pred, axis=1))
        pred_y = np.concatenate(pred_y)
        print ('Testing Accuracy: {}'.format(accuracy_score(test_y, pred_y)))
        print (classification_report(test_y, pred_y))

    # return the query function
    if rtn_layer:
        return output_layer
    else:
        return pred_y

def load_dataset1(train_feat, train_label, test_feat=None, test_label=None):
    train_x = np.genfromtxt(train_feat, delimiter=',', dtype='float32')
    train_y = np.genfromtxt(train_label, dtype='int32')
    min_y = np.min(train_y)
    train_y -= min_y
    if test_feat is not None and test_label is not None:
        test_x = np.genfromtxt(train_feat, delimiter=',', dtype='float32')
        test_y = np.genfromtxt(train_label, dtype='int32')
        test_y -= min_y
    else:
        test_x = None
        test_y = None
    return train_x, train_y, test_x, test_y


def load_dataset(train_feat, train_label, test_feat=None, test_label=None):
    train_x = np.genfromtxt(train_feat, delimiter=',', dtype='float32')
    train_label=pd.read_csv('train_label.csv')
    y_train = []
    for i in range(train_x.shape[0]):
        y_train.append(train_label.iloc[i]['label'])

    y_train = np.asarray(y_train)
    train_y = y_train
    train_y=np.asarray(train_y)
    train_y=train_y.astype('int32')
    print(np.unique(train_y))
    print(train_y.shape)
    print(train_y.shape)
    if test_feat is not None and test_label is not None:
        test_x = np.genfromtxt(train_feat, delimiter=',', dtype='float32')
        test_y = np.genfromtxt(train_label, dtype='int32')
        test_y -= min_y
    else:
        test_x = None
        test_y = None
    return train_x, train_y, test_x, test_y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_feat', type=str)
    parser.add_argument('train_label', type=str)
    parser.add_argument('--test_feat', type=str, default=None)
    parser.add_argument('--test_label', type=str, default=None)
    parser.add_argument('--model', type=str, default='nn')
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--n_hidden', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    print (vars(args))
    dataset = load_dataset(args.train_feat1, args.train_label, args.test_feat, args.train_label)
    train(dataset,
          model=args.model,
          learning_rate=args.learning_rate,
          batch_size=args.batch_size,
          n_hidden=args.n_hidden,
          epochs=args.epochs)


if __name__ == '__main__':
    main()
