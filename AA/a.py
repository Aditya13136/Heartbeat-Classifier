# ============================================
# Binary ECG Heartbeat Classifier (Normal vs Abnormal)
# ============================================

import wfdb
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# ============================================
# Step 1: Load Multiple Records
# ============================================
data_path = "C:/Users/HP/Documents/GitHub/A/AA/mit-bih-arrhythmia-database-1.0.0/"

records = ['100','101','102','103','104','105','106','107','108','109']

window_size = 100

X = []
y = []

# Binary label mapping
label_map = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,   # Normal
    'V': 1, 'E': 1, 'A': 1, 'a': 1,
    'F': 1, '/': 1, 'f': 1, 'Q': 1            # Abnormal
}

print("Loading data...")

for rec in records:
    print(f"Processing record {rec}...")
    
    record = wfdb.rdrecord(data_path + rec)
    annotation = wfdb.rdann(data_path + rec, 'atr')
    
    signal = record.p_signal[:, 0]
    r_peaks = annotation.sample
    labels = annotation.symbol
    
    for i in range(len(r_peaks)):
        peak = r_peaks[i]
        
        if peak - window_size < 0 or peak + window_size > len(signal):
            continue
        
        if labels[i] not in label_map:
            continue
        
        beat = signal[peak - window_size : peak + window_size]
        label = label_map[labels[i]]
        
        X.append(beat)
        y.append(label)

X = np.array(X)
y = np.array(y)

print(f"Total beats: {len(X)}")

# ============================================
# Step 2: Shuffle
# ============================================
X, y = shuffle(X, y, random_state=42)

# ============================================
# Step 3: Normalize & Reshape
# ============================================
scaler = StandardScaler()
X = scaler.fit_transform(X)

X = X.reshape(X.shape[0], X.shape[1], 1)

# One-hot encoding
y = to_categorical(y, num_classes=2)

# ============================================
# Step 4: Train-Test Split
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================
# Step 5: Handle Class Imbalance
# ============================================
y_labels = np.argmax(y_train, axis=1)

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_labels),
    y=y_labels
)

class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

# ============================================
# Step 6: Build CNN Model
# ============================================
model = Sequential([
    Conv1D(32, 5, activation='relu', input_shape=(X.shape[1], 1)),
    MaxPooling1D(2),

    Conv1D(64, 3, activation='relu'),
    MaxPooling1D(2),

    Conv1D(128, 3, activation='relu'),
    MaxPooling1D(2),

    Dropout(0.5),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')   # ONLY 2 OUTPUTS
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ============================================
# Step 7: Train Model
# ============================================
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=32,
    class_weight=class_weights
)

# ============================================
# Step 8: Evaluate
# ============================================
loss, acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {acc*100:.2f}%")

# ============================================
# Step 9: Predict from TEST data (unseen)
# ============================================
sample = X_test[0].reshape(1, X.shape[1], 1)

prediction = model.predict(sample)
pred_class = np.argmax(prediction)
confidence = np.max(prediction)

class_names = ['Normal', 'Abnormal']

print(f"Predicted: {class_names[pred_class]} ({confidence*100:.2f}%)")

# ============================================
# Step 10: Predict from NEW RECORD (real new data)
# ============================================
print("\nTesting on completely new record...")

record = wfdb.rdrecord(data_path + '116')
annotation = wfdb.rdann(data_path + '116', 'atr')

signal = record.p_signal[:, 0]
peak = annotation.sample[0]

new_beat = signal[peak - window_size : peak + window_size]

# Apply SAME preprocessing
new_beat = scaler.transform([new_beat])
new_beat = new_beat.reshape(1, new_beat.shape[1], 1)

prediction = model.predict(new_beat)
pred_class = np.argmax(prediction)
confidence = np.max(prediction)

print(f"New Beat Prediction: {class_names[pred_class]} ({confidence*100:.2f}%)")

# ============================================
# Step 11: Plot Training Graph
# ============================================
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.show()