# 🩻 AI-Based Fracture Detection Web App  
### Deep Learning Project using MURA Dataset

---

<p align="center">
  <img src="https://img.shields.io/badge/Framework-Flask-blue?style=for-the-badge">
  <img src="https://img.shields.io/badge/Model-TensorFlow-orange?style=for-the-badge">
  <img src="https://img.shields.io/badge/Type-DeepLearning-green?style=for-the-badge">
  <img src="https://img.shields.io/badge/Status-Completed-success?style=for-the-badge">
</p>

---

## 📌 Overview

This project is a **deep learning-based web application** that detects bone fractures from X-ray images.  
It leverages the **MURA (Musculoskeletal Radiographs)** dataset and a **DenseNet121 model** to classify images into:

- 🟢 **Normal (Negative)**
- 🔴 **Abnormal (Positive)**

The trained model is integrated into a **Flask web application**, enabling real-time predictions via image upload.

---

## 🎯 Objectives

- Build an accurate fracture detection model using deep learning  
- Deploy the model as a user-friendly web application  
- Provide real-time predictions for medical image analysis  

---

## 🧠 Technologies Used

| Category            | Tools/Technologies              |
|--------------------|--------------------------------|
| 🧠 Deep Learning   | TensorFlow, Keras              |
| 🌐 Web Framework   | Flask                          |
| 🖼 Image Handling  | Pillow, NumPy                  |
| 🤖 AI Assistance   | Claude AI                      |
| 💻 Language        | Python                         |

---

## 🏗 Model Architecture

- **Base Model:** DenseNet121 (Pre-trained on ImageNet)
- **Transfer Learning:** Yes  
- **Input Size:** 224 × 224 × 3  
- **Output:** Binary Classification  
- **Activation:** Sigmoid  
- **Loss Function:** Binary Crossentropy  
- **Optimizer:** Adam  

---

## 📂 Project Structure
