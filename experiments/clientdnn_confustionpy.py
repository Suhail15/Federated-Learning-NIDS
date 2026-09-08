
import flwr as fl
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers.legacy import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay




def load_data(filename):
    global X, y, df
    df = pd.read_csv(filename)
    X = df.iloc[:, :-1].values  # Features: all columns except the last one
    y = df.iloc[:, -1].values  # Labels: the last column

    return X, y 


def build_dnn_model(output_units=5):  # Adjust `output_units` based on your specific task
    model = tf.keras.Sequential([
        # Input layer with increased units and input shape for 24 features
        tf.keras.layers.Dense(512, activation='relu', input_shape=(23,)),
        tf.keras.layers.Dropout(0.2),  # Increased dropout for regularization
        
        # Added additional dense layers with increased units
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),  # Additional dropout layer for regularization
        
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.1),  # Keep dropout to prevent overfitting
        
        # Additional layer to increase model depth
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.1),  # Keep dropout to prevent overfitting

        tf.keras.layers.Dense(64, activation='relu'),
        
        # Output layer for multi-class classification with softmax activation
        tf.keras.layers.Dense(output_units, activation='softmax')
    ])
    return model

# Assuming load_data function and DNNClient class are defined as in your previous snippets
X, y = load_data('data/df_aug_shuff_c1.csv')  # Load your dataset

# Split the dataset into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
num_classes = 5  # Update this based on your task
model = build_dnn_model(output_units=num_classes)
model.compile(optimizer=Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=128, epochs=10, verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(x_val, y_val, verbose=0)
print(f"Validation loss: {loss}, Validation accuracy: {accuracy}")

# Optional: Generate predictions and display the confusion matrix
predictions = model.predict(x_val)
predicted_classes = np.argmax(predictions, axis=1)
cm = confusion_matrix(y_val, predicted_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Confusion Matrix')
plt.show()



