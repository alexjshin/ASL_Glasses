import os
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from preprocess_data import preprocess_data

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

X_train, X_test, y_train, y_test, actions = preprocess_data('MP_Data_01')

# # Define input shape (sequence_length, features)
# inputs = Input(shape=(30, 1662))

# # First LSTM layer
# x = LSTM(64, return_sequences=True, activation='relu')(inputs)

# # Second LSTM layer
# x = LSTM(128, return_sequences=True, activation='relu')(x)

# # Third LSTM layer
# x = LSTM(64, return_sequences=False, activation='relu')(x)

# # Dense layers
# x = Dense(64, activation='relu')(x)
# x = Dense(32, activation='relu')(x)

# # Output layer
# outputs = Dense(actions.shape[0], activation='softmax')(x)

# # Create model
# model = Model(inputs=inputs, outputs=outputs)

# SIMPLE LSTM MODEL (for full landmarks)
# inputs = Input(shape=(30, 1662))
# x = LSTM(32, return_sequences=False, activation='relu')(inputs)
# x = Dense(32, activation='relu')(x)
# outputs = Dense(actions.shape[0], activation='softmax')(x)

# New LSTM Model (for hand landmarks only)
inputs = Input(shape=(30, 258))
x = LSTM(64, return_sequences=True, activation='relu')(inputs)
x = Dropout(0.2)(x)
x = LSTM(128, return_sequences=False, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.2)(x)
outputs = Dense(actions.shape[0], activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback])

model.save('Models/02_hand_pose_lstm_model.h5')
model.save_weights('Models/02_hand_pose_model.weights.h5')

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")
print(f"Test loss: {test_loss:.4f}")