# Handwritten Digit Recognition using Logistic Regression

This project builds a **multi-class image classification system** to recognize handwritten digits using **Logistic Regression** trained on the MNIST dataset. The model predicts digits (0–9) based on image input.

## 📌 Project Objective

Develop a machine learning model that can classify handwritten digit images (multi-class classification). You can upload your own digit images, and the model will predict the correct digit.

## 🔧 Technologies Used

- Python
- Scikit-learn
- OpenCV
- NumPy
- Matplotlib
- Jupyter Notebook / Google Colab

## 📂 Project Structure

The project is organized into the following steps (cells in the notebook):

1. **Import Libraries**  
   Import necessary Python libraries like `sklearn`, `numpy`, `cv2`, etc.

2. **Load and Preprocess the MNIST Dataset**  
   Load digit images from the MNIST dataset, normalize pixel values, and scale features.

3. **Train-Test Split**  
   Split the data into training and test sets.

4. **Train Logistic Regression Model**  
   Fit a multi-class logistic regression model on the training set.

5. **Evaluate the Model**  
   Test the model on unseen data and calculate accuracy.

6. **Preprocess Custom Image Input**  
   Process your uploaded handwritten digit image:
   - Convert to grayscale
   - Center and resize to 28×28
   - Normalize and scale

7. **Make Predictions**  
   Use the trained model to predict the digit in your uploaded image.

## 🖼️ Custom Image Requirements

- Upload a **clear black digit on a white background**
- Format: `.png` or `.jpg`
- The digit should be **centered** and **not too small**
- Images are automatically resized to 28×28

## ▶️ How to Run

1. Clone or download the project
2. Open the notebook in **Jupyter** or **Google Colab**
3. Run each cell in order
4. Upload your own digit image in the correct cell
5. Check the predicted result printed below the image

## 📈 Model Used

- **Logistic Regression** with `multi_class='ovr'`
- Trained on ~60,000 MNIST images
- Uses feature scaling (`StandardScaler`) for better convergence

## ⚠️ Notes

- This is a basic model. For better performance, consider using **CNNs (Convolutional Neural Networks)**.
- Logistic Regression may not perform well if images are noisy, low-contrast, or poorly written.

## 🙋 Author

This project was developed as part of a formative assessment. All code and descriptions are written originally by the student, assisted by understanding concepts through experimentation and research.
