from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout,MaxPooling2D
from tensorflow.keras.models import Sequential
import tensorflow as tf

# Load and preprocess training data
(train_X, train_y), (test_X, test_y) = mnist.load_data()
train_X, test_X = map(lambda X: (X.astype('float') / 255.).reshape((-1, 28, 28, 1)), [train_X, test_X])
train_y, test_y = map(lambda X: to_categorical(X, 10), [train_y, test_y])

# Hyperparameters
BATCH_SIZE = 128
EPOCHS = 50
optimizer = Adam(lr=1e-3)

# Build model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer=RandomNormal(), input_shape=(28, 28, 1)))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer=RandomNormal()))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer=RandomNormal()))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax', kernel_initializer=RandomNormal()))

def step(X, y):
    with tf.GradientTape() as tape:
        # Predict
        preds = model(X)

        # Calculate loss
        loss = tf.keras.losses.categorical_crossentropy(y, preds)

    # Calculate the gradients
    grads = tape.gradient(loss, model.trainable_variables)

    # Update model
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

# Training loop
for epoch in range(EPOCHS):
    for i in range(0, len(train_y), BATCH_SIZE):
        batch_X, batch_y = train_X[i:i + BATCH_SIZE], train_y[i:i + BATCH_SIZE]
        step(batch_X, batch_y)

# Calculate accuracy
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
print(model.evaluate(test_X, test_y, verbose=0)) # [0.07158771060718329, 0.9911]