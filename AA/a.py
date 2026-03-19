# cnn_heartbeat_classifier.py

import wfdb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# ===============================
# Step 1: Load MIT-BIH Record
# ===============================
record = wfdb.rdrecord('C:/Users/HP/Documents/GitHub/A/AA/mit-bih-arrhythmia-database-1.0.0/101')
annotation = wfdb.rdann('C:/Users/HP/Documents/GitHub/A/AA/mit-bih-arrhythmia-database-1.0.0/101', 'atr')

signal = record.p_signal[:, 0]  # use first ECG channel
r_peaks = annotation.sample
labels = annotation.symbol

# ===============================
# Step 2: Extract Heartbeat Segments
# ===============================
window_size = 100  # samples before and after R-peak
X = []
y = []

for i in range(len(r_peaks)):
    peak = r_peaks[i]
    
    if peak - window_size < 0 or peak + window_size > len(signal):
        continue
    
    beat = signal[peak - window_size : peak + window_size]
    
    # Map labels to binary: Normal (N) vs Abnormal (others)
    label = 0 if labels[i] == 'N' else 1
    
    X.append(beat)
    y.append(label)

X = np.array(X)
y = np.array(y)

print(f"Total beats extracted: {len(X)}")

# ===============================
# Step 3: Normalize & Reshape
# ===============================
scaler = StandardScaler()
X = scaler.fit_transform(X)

# CNN expects 3D input: (samples, timesteps, channels)
X = X.reshape(X.shape[0], X.shape[1], 1)

# One-hot encode labels
y = to_categorical(y, num_classes=2)

# ===============================
# Step 4: Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===============================
# Step 5: Build CNN Model
# ===============================
model = Sequential([
    Conv1D(filters=32, kernel_size=5, activation='relu', input_shape=(X.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')  # 2 classes: normal, abnormal
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ===============================
# Step 6: Train the Model
# ===============================
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=32
)

# ===============================
# Step 7: Evaluate Model
# ===============================
loss, acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {acc*100:.2f}%")

# ===============================
# Step 8: Predict a New Heartbeat
# ===============================
sample = X_test[0].reshape(1, X.shape[1], 1)
prediction = model.predict(sample)
pred_label = np.argmax(prediction, axis=1)[0]

print("Predicted:", "Normal" if pred_label == 0 else "Abnormal")

# ===============================
# Step 9: Optional: Visualize Training History
# ===============================
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()