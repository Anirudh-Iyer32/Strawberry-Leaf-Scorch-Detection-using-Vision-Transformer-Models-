# Strawberry Leaf Scorch Disease Detection using Lightweight Vision Transformer Models

## Project Overview
Leaf scorch is a common and destructive disease affecting strawberry crops, leading to reduced yield and crop quality. This project focuses on **early and accurate detection of strawberry leaf scorch disease** using **lightweight Vision Transformer (ViT) architectures**, leveraging recent advances in transformer-based image classification.

Unlike traditional CNN-based approaches, this project explores **transformer models optimized for efficiency**, making them suitable for real-world and resource-constrained agricultural applications.

---

## Objectives
- Apply modern **Vision Transformer architectures** for plant disease classification  
- Evaluate and compare multiple **lightweight ViT models**  
- Test model performance on **real-time, field-collected images**  
- Develop an **ensemble model** to improve robustness and accuracy  
- Demonstrate the effectiveness of transformers in agricultural disease detection  

---

## Dataset
- The dataset consists of strawberry leaf images categorized into:
  - Healthy leaves
  - Leaf scorch infected leaves
- Images were preprocessed and augmented to improve generalization
- Additional **real-time images** were collected and used for external validation to assess real-world performance

---

## Methodology

### Image Preprocessing
- Image resizing and normalization
- Data augmentation to reduce overfitting
- Train-validation-test split for fair evaluation

---

## Vision Transformer Models Used
This project utilizes four transformer-based architectures, focusing on efficiency and performance:

### Lightweight Vision Transformers
- **MobileViT** – Combines convolutional layers with transformer blocks for mobile-friendly performance  
- **NextViT** – A hybrid architecture designed for fast inference and high accuracy  
- **MicroViT** – Optimized for low-parameter and low-compute environments  

### Traditional Transformer
- **Efficient Transformer** – Used as a baseline transformer model for comparison  

Each model was trained independently and evaluated on both test data and real-time images.

---

## Ensemble Learning
To further improve performance:
- An **ensemble model** was developed by combining predictions from the three lightweight ViT models:
  - MobileViT
  - NextViT
  - MicroViT
- Ensemble predictions were obtained using aggregation techniques
- This approach improved stability and generalization compared to individual models

---

## Model Evaluation
Models were evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Special emphasis was placed on **real-time image performance** to validate practical usability.

---

## Real-Time Testing
- The trained models were tested on **real-world strawberry leaf images** collected separately from the training dataset  
- This step ensured:
  - Robustness against environmental variations
  - Better assessment of field-level deployment readiness  

---

## Technologies Used
- Python
- PyTorch / TensorFlow (as applicable)
- Vision Transformer architectures
- NumPy
- OpenCV
- Matplotlib
- Scikit-learn

---

## Key Highlights
- Application of **transformer-based models** for plant disease detection  
- Use of **lightweight ViT architectures** suitable for edge and mobile devices  
- Real-time data validation for practical performance assessment  
- Ensemble learning to enhance prediction accuracy  
- Demonstrates the shift from CNNs to **transformers in image classification**

---

## Future Scope
- Deployment on mobile or edge devices for farmers
- Expansion to multi-disease and multi-crop classification
- Integration with drone or IoT-based image capture systems
- Optimization for real-time inference

---

## Disclaimer
This project is intended for academic and research purposes only.  
It does not replace professional agricultural consultation or disease management practices.
