from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import KFold
import numpy as np

# Model configuration
batch_size = 50
no_classes = 100
no_epochs = 25
num_folds = 10

# Define per-fold score containers
accuracy_list = []
loss_list = []

def load_data():

  # Load dữ liệu CIFAR đã được tích hợp sẵn trong Keras
  (X_train, y_train), (X_test, y_test) = cifar10.load_data()

  # Chuẩn hoá dữ liệu
  X_train = X_train.astype('float32')
  X_test = X_test.astype('float32')
  X_test = X_test / 255
  X_train = X_train / 255

  # Do CIFAR đã chia sẵn train và test nên ta nối lại để chia K-Fold
  X = np.concatenate((X_train, X_test), axis=0)
  y = np.concatenate((y_train, y_test), axis=0)

  return X, y


def get_model():

  model = Sequential()
  model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Flatten())
  model.add(Dense(256, activation='relu'))
  model.add(Dense(128, activation='relu'))
  model.add(Dense(no_classes, activation='softmax'))

  # Compile  model
  model.compile(loss="sparse_categorical_crossentropy",
                optimizer="Adam",
                metrics=['accuracy'])

  return model



X, y = load_data()

# Định nghĩa K-Fold CV
kfold = KFold(n_splits=num_folds, shuffle=True)

# K-fold Cross Validation model evaluation
fold_idx = 1

for train_ids, val_ids in kfold.split(X, y):

  model = get_model()

  print("Bắt đầu train Fold ", fold_idx)

  # Train model
  model.fit(X[train_ids], y[train_ids],
              batch_size=batch_size,
              epochs=no_epochs,
              verbose=1)

  # Test và in kết quả
  scores = model.evaluate(X[val_ids], y[val_ids], verbose=0)
  print("Đã train xong Fold ", fold_idx)

  # Thêm thông tin accuracy và loss vào list
  accuracy_list.append(scores[1] * 100)
  loss_list.append(scores[0])

  # Sang Fold tiếp theo
  fold_idx = fold_idx + 1


# In kết quả tổng thể

print('* Chi tiết các fold')
for i in range(0, len(accuracy_list)):
  print(f'> Fold {i+1} - Loss: {loss_list[i]} - Accuracy: {accuracy_list[i]}%')

print('* Đánh giá tổng thể các folds:')
print(f'> Accuracy: {np.mean(accuracy_list)} (Độ lệch +- {np.std(accuracy_list)})')
print(f'> Loss: {np.mean(loss_list)}')