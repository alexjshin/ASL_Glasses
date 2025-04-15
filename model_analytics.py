import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from preprocess_data import preprocess_data

model = load_model('Models/02_lstm_model.h5')
model.load_weights('Models/02_model.weights.h5')

X_train, X_test, y_train, y_test, actions = preprocess_data('MP_Data_01', sequence_length=30)

######## EVALUATE USING CONFUSION MATRIX ########
yhat = model.predict(X_test)
ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()
print(multilabel_confusion_matrix(ytrue, yhat))
print(accuracy_score(ytrue, yhat))