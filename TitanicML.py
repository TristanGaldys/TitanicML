from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.regularizers import L2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout, BatchNormalization
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import tensorflow as tf
import matplotlib.pyplot as plt

def plot(history):
    plt.plot(history.history['accuracy'])
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

def preprocess(file):
    train_df = pd.read_csv(file)  # Adjust the path to where you've uploaded your train.csv

    # Select features and target
    X = train_df.drop(['Survived', 'Name', 'Ticket', 'Cabin'], axis=1)  # Exclude non-numeric columns for simplicity
    y = train_df['Survived']
    # Fill missing values
    # For numerical columns
    num_imputer = SimpleImputer(strategy='mean')
    # For categorical columns
    cat_imputer = SimpleImputer(strategy='most_frequent')

    # Encode categorical data and scale numerical data
    ohe = OneHotEncoder()
    scaler = StandardScaler()

    # Make column transformer
    preprocessor = make_column_transformer(
        (make_pipeline(num_imputer, scaler), ['Age', 'Fare']),
        (make_pipeline(cat_imputer, ohe), ['Pclass', 'Sex', 'Embarked']),
        remainder='passthrough'  # Include other columns without transformations
    )

    # Fit the preprocessor and transform the data
    X_processed = preprocessor.fit_transform(X)
    # Convert to TensorFlow tensors
    X_train = tf.convert_to_tensor(X_processed, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y, dtype=tf.float32)
    return (X_train, y_train)

# Load your data
X, y = preprocess('train.csv')

# Define batch_size and training_epochs
batch_size = 32
training_epochs = 2500

# Build the model
model = Sequential([
    Input(shape=(X.shape[1],)),
    Dense(16, activation='relu', kernel_regularizer=L2(0.01)),
    BatchNormalization(),
    Dense(12, activation='relu', kernel_regularizer=L2(0.01)),
    BatchNormalization(),
    Dense(8, activation='relu', kernel_regularizer=L2(0.01)),
    BatchNormalization(),
    Dense(4, activation='relu', kernel_regularizer=L2(0.01)),
    Dense(1, activation='sigmoid')
])
adam = Adam(learning_rate=0.00005)  # Adjust the learning rate as needed
# Compile the model
model.compile(optimizer=adam , loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
plot(model.fit(X, y, batch_size=batch_size, epochs=training_epochs))

X_test, y_test = preprocess('test.csv')

test_loss, test_accuracy = model.evaluate(X_test, y_test, batch_size=batch_size)

print(test_loss)
print(test_accuracy)


