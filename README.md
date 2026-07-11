# Task 03 - SVM Image Classifier (Cats vs Dogs)

## Objective
Implement a Support Vector Machine (SVM) to classify images of cats and dogs, using the Kaggle "Dogs vs Cats" dataset.

## Dataset
- Source: Kaggle "Dogs vs Cats" dataset (25,000 labeled images)
- Used a subset of 2,000 images (1,000 cats + 1,000 dogs) to keep training time manageable
- Images resized to 64x64 pixels and flattened into numeric arrays for the model

## Approach
1. Loaded and resized images using OpenCV
2. Flattened each image into a 1D array of pixel values
3. Scaled the pixel values using `StandardScaler` for better model performance
4. Split data into 80% training / 20% testing
5. Trained an SVM classifier (`sklearn.svm.SVC`) using an RBF kernel

## Results
- **Accuracy: 63.25%**
- Precision/Recall are fairly balanced between cats and dogs (~62-64% each)
- Initial attempt with a linear kernel and unscaled pixels achieved only 54.75% accuracy — scaling the data and switching to an RBF kernel improved performance noticeably

## Notes
- Accuracy is modest because this uses raw pixel values rather than more advanced feature extraction (e.g. deep learning/CNNs), which is expected for a classical SVM approach on image data
-