import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Bước 1: Đọc file CSV chứa các đặc trưng HOG và nhãn
input_csv = 'Output/hog.csv'  # Đường dẫn tới tệp CSV
df = pd.read_csv(input_csv)

# Bước 2: Chuẩn bị dữ liệu
# Giả định rằng df là DataFrame chứa dữ liệu HOG và nhãn
hog_features = df.iloc[:, 1:513]  # Các cột đặc trưng HOG (cột 2 đến cột 513)
labels = df.iloc[:, 513]          # Cột nhãn (cột 514)

# Hiển thị một số giá trị đầu và cuối của hog_features
print("Giá trị đầu của hog_features:")
print(hog_features.head())

print("\nGiá trị cuối của hog_features:")
print(hog_features.tail())

print("\nGiá trị cuối của nhãn:")
print(labels.head())


# Hiển thị giá trị đầu và cuối của đặc trưng HOG cho mỗi hàng

X = hog_features.values  # Chuyển các đặc trưng HOG thành mảng numpy
y = labels.values        # Chuyển nhãn thành mảng numpy

# Bước 3: Chia dữ liệu theo phương pháp K-Fold với k=5
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Đánh giá mô hình KNN với k=3
accuracies_k3 = []
print("Đánh giá mô hình KNN với k=3:")
for fold, (train_index, test_index) in enumerate(kf.split(X)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    print(f"Fold {fold + 1}:")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=1))
    accuracy = (y_pred == y_test).mean()
    print(f"Độ chính xác: {accuracy:.2f}\n")

    accuracies_k3.append(accuracy)

# Tính toán độ chính xác trung bình cho k=3
avg_accuracy_k3 = np.mean(accuracies_k3)
print(f"Độ chính xác trung bình cho k=3: {avg_accuracy_k3:.2f}\n")

# Đánh giá mô hình KNN với k=5
accuracies_k5 = []
print("Đánh giá mô hình KNN với k=5:")
for fold, (train_index, test_index) in enumerate(kf.split(X)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    print(f"Fold {fold + 1}:")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=1))
    accuracy = (y_pred == y_test).mean()
    print(f"Độ chính xác: {accuracy:.2f}\n")

    accuracies_k5.append(accuracy)

# Tính toán độ chính xác trung bình cho k=5
avg_accuracy_k5 = np.mean(accuracies_k5)
print(f"Độ chính xác trung bình cho k=5: {avg_accuracy_k5:.2f}\n")
