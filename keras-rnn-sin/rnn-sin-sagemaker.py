from keras.models import Sequential  
from keras.layers.core import Dense, Activation  
from keras.layers.recurrent import LSTM
from keras import backend as K
import numpy as np
import tensorflow as tf
import argparse
import os

in_out_neurons = 1
hidden_neurons = 300
length_of_sequences = 100

def train(args):
    batch_size = 600
    epochs = 15
    validation_split = 0.05
    
    # SageMakerのトレーニングに使われるコンテナは起動時に、inputで指定したS3からデータをダウンロードし以下のパスに保存します
    channel_input_dirs = '/opt/ml/input/data/training/'
    
    # トレーニング用データを読み込みます
    X_train = np.load(channel_input_dirs + 'X_train.npy')
    y_train = np.load(channel_input_dirs + 'y_train.npy')
    
    model = Sequential()  
    model.add(LSTM(hidden_neurons, batch_input_shape=(None, length_of_sequences, in_out_neurons), return_sequences=False))  
    model.add(Dense(in_out_neurons))  
    model.add(Activation("linear"))  
    model.compile(loss="mean_squared_error", optimizer="rmsprop")
    
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split,verbose=1)

    save(model, args.model_dir)
    

def save(model, model_dir):
    sess = K.get_session()
    tf.saved_model.simple_save(
        sess,
        os.path.join(model_dir, 'model/1'),
        inputs={'inputs': model.input},
        outputs={t.name: t for t in model.outputs})
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument('--batch-size', type=int, default=600)
    parser.add_argument('--validation_split', type=float , default=0.05)
    parser.add_argument('--epochs', type=int, default=15)

    # input data and model directories
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAINING'])

    args, _ = parser.parse_known_args()
    train(args)