import os
import cv2
import numpy as np

# Path to the folder containing all the images
data_folder = "train/train"

# We'll only use a subset to keep things fast
num_images_per_class = 1000

images = []
labels = []

for i in range(num_images_per_class):
    for animal in ["cat", "dog"]:
        filename = f"{animal}.{i}.jpg"
        filepath = os.path.join(data_folder, filename)

        img = cv2.imread(filepath)          # read the image
        img = cv2.resize(img, (64, 64))     # shrink it to 64x64 pixels
        images.append(img)
        labels.append(0 if animal == "cat" else 1)  # cat = 0, dog = 1

print("Loaded", len(images), "images")
print("Example label:", labels[0])
# Convert lists into NumPy arrays (a format ML tools expect)
images = np.array(images)
labels = np.array(labels)

# Flatten each image: turn the 64x64 grid of color pixels into one long list of numbers
images = images.reshape(len(images), -1)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
images = scaler.fit_transform(images)

print("Shape of images array:", images.shape)
from sklearn.model_selection import train_test_split

# Split data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=42
)

print("Training images:", X_train.shape[0])
print("Testing images:", X_test.shape[0])

from sklearn.svm import SVC
import time

# Create the SVM model
model = SVC(kernel='rbf')

print("Training the model... this may take a minute or two.")
start_time = time.time()

model.fit(X_train, y_train)

end_time = time.time()
print(f"Training finished in {end_time - start_time:.1f} seconds")

from sklearn.metrics import accuracy_score, classification_report

# Use the trained model to predict labels for the test images
predictions = model.predict(X_test)

# Compare predictions to the actual correct labels
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nDetailed report:")
print(classification_report(y_test, predictions, target_names=["cat", "dog"]))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["cat", "dog"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Cat vs Dog Classification")
plt.savefig("confusion_matrix.png")
print("Confusion matrix saved as confusion_matrix.png")