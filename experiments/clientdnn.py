import flwr as fl
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers.legacy import Adam

def build_dnn_model(output_units=5):  
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation='relu', input_shape=(23,)),
        tf.keras.layers.Dropout(0.2),   
        
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2), 
        
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.1),  
        
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.1),  

        tf.keras.layers.Dense(64, activation='relu'),
        
        tf.keras.layers.Dense(output_units, activation='softmax')
    ])
    return model


def load_data(filename):
    df = pd.read_csv(filename)
    X = df.iloc[:, :-1].values 
    y = df.iloc[:, -1].values  

    return X, y 

class DNNClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train

    def get_parameters(self, **kwargs):
        return self.model.get_weights()

    def set_parameters(self, parameters):
        self.model.set_weights(parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        batch_size = config.get("batch_size", 128)  
        epochs = config.get("epochs", 10)  
        self.model.fit(self.x_train, self.y_train, batch_size=batch_size, epochs=epochs, verbose=0)
        return self.get_parameters(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = self.model.evaluate(self.x_train, self.y_train, verbose=0)
        print(f"Test loss: {loss}, Test accuracy: {accuracy}")
        return loss, len(self.x_train), {"accuracy": accuracy}

if __name__ == "__main__":
    num_clients = 2

    full_train_data, full_train_labels= load_data('data/df_aug_shuff_c1.csv') 
    chunk_size = len(full_train_data) // num_clients
    x_chunks = [full_train_data[i:i + chunk_size] for i in range(0, len(full_train_data), chunk_size)]
    y_chunks = [full_train_labels[i:i + chunk_size] for i in range(0, len(full_train_labels), chunk_size)]

    x_train = x_chunks[0]  
    y_train = y_chunks[0]  

    num_classes = 5  
    model = build_dnn_model(output_units=num_classes)
    model.compile(optimizer=Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])  # Adjust loss and metrics

    client = DNNClient(model, x_train, y_train)
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)
 

