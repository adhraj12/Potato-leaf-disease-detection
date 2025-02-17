# Potato Leaf Disease Detection

## Overview
Potato leaf diseases, such as early blight and late blight, significantly impact potato crop yields and farmer livelihoods. Traditional disease detection methods are labor-intensive, time-consuming, and prone to human error. This project introduces an AI-powered solution utilizing deep learning techniques to automate potato leaf disease detection.

## Features
- **Deep Learning-Based Classification**: Uses a Convolutional Neural Network (CNN) model.
- **Transfer Learning**: Employs the ResNet50 model, fine-tuned for disease detection.
- **High Accuracy**: Achieves 99% accuracy on the test set.
- **Web Application**: Built using Streamlit for easy image uploads and real-time predictions.
- **Open Source**: Code and model available for further research and development.

## Dataset
The dataset consists of potato leaf images categorized into:
- Healthy
- Early Blight
- Late Blight

Data augmentation techniques, such as rotations, flips, and brightness adjustments, were applied to improve model generalization.

## System Architecture
1. **Image Acquisition**: Images of potato leaves are collected.
2. **Preprocessing**: Images are resized, normalized, and augmented.
3. **Feature Extraction**: ResNet50 extracts meaningful features.
4. **Classification**: A fully connected layer classifies images into three categories.
5. **Prediction Output**: The model provides a disease classification with confidence scores.

## Technical Requirements
### Hardware
- **Processor**: Intel Core i5 or AMD Ryzen 5 (or higher recommended)
- **RAM**: 8GB (16GB recommended)
- **GPU**: NVIDIA GPU with CUDA support (e.g., GTX 1050 Ti or higher)
- **Storage**: At least 20GB free disk space

### Software
- **Operating System**: Windows 10/11, Linux (Ubuntu), macOS
- **Programming Language**: Python 3.7+
- **Deep Learning Framework**: TensorFlow 2.x or PyTorch 1.x+
- **Libraries**:
  - Keras (if using TensorFlow)
  - NumPy
  - Matplotlib
  - Scikit-learn
  - Streamlit (for web app deployment)

## Model Training
The model was trained using the Adam optimizer with:
- **Learning Rate**: 0.001
- **Batch Size**: 32
- **Epochs**: 20
- **Dropout Rate**: 0.5
- **Image Size**: 224x224

## Performance Metrics
| Metric      | Value  |
|------------|--------|
| Accuracy   | 99%    |
| Precision  | ~98-99%|
| Recall     | ~98-99%|
| F1-Score   | ~98-99%|

## Web Application
The model is deployed as a **Streamlit** web application that allows users to upload images and receive disease classification results in real-time.

## Future Work
- Expanding the dataset to include more potato varieties and real-world images.
- Extending classification to additional potato diseases.
- Enhancing model architecture with advanced CNNs like EfficientNet.
- Deploying a mobile-friendly version using TensorFlow Lite.
- Integrating with drones for large-scale crop monitoring.
- Implementing disease severity estimation and treatment recommendations.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/adhraj12/Potato-leaf-disease-detection.git
   cd Potato-leaf-disease-detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit web app:
   ```bash
   streamlit run app.py
   ```
4. Upload an image of a potato leaf to get real-time predictions.

## Contributing
Contributions are welcome! Feel free to fork the repository and submit pull requests.

## License
This project is licensed under the MIT License.

## Contact
For any queries, contact:
- **Adhiraj Jagtap** - [adhirajjagtap12@gmail.com](mailto:adhirajjagtap12@gmail.com)



