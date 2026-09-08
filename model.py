"""Original capstone DNN: 23 features and five output classes."""
import tensorflow as tf

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
