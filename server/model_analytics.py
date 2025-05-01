import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from preprocess_data import preprocess_data
import seaborn as sns

model = load_model('../Models/02_hand_pose_lstm_model.h5')
model.load_weights('../Models/02_hand_pose_model.weights.h5')

X_train, X_test, y_train, y_test, actions = preprocess_data('../MP_Data_01', sequence_length=30)

######## EVALUATE USING CONFUSION MATRIX ########
# yhat = model.predict(X_test)
# ytrue = np.argmax(y_test, axis=1).tolist()
# yhat = np.argmax(yhat, axis=1).tolist()
# print(multilabel_confusion_matrix(ytrue, yhat))
# print(accuracy_score(ytrue, yhat))

# Make predictions
yhat = model.predict(X_test)
ytrue = np.argmax(y_test, axis=1)
yhat = np.argmax(yhat, axis=1)

# Compute the confusion matrix
cm = confusion_matrix(ytrue, yhat)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Create a figure
plt.figure(figsize=(10, 8))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
            xticklabels=actions, yticklabels=actions)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for ASL Sign Recognition')
plt.tight_layout()

# Save the figure
plt.savefig('confusion_matrix.png', dpi=300)
plt.show()

# Print overall accuracy
print(f"Overall accuracy: {accuracy_score(ytrue, yhat):.4f}")