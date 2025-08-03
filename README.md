# 🧪 Biomedical Waste Classification using Edge AI and Deep Learning

A real-time, intelligent biomedical waste classification system using **Vision Transformers (ViT)**, **YOLOv11**, **EfficientDet-Lite0**, and **Custom CNN**, deployed on edge devices like **Raspberry Pi**. The system aims to automate and optimize medical waste segregation for improved public health and environmental safety.

## 🚀 Overview

Improper segregation of biomedical waste is a major threat to public health and the environment. This project presents an **Edge-AI powered Smart Bin** that classifies biomedical waste into categories like:

- 🔪 Sharps  
- 🧫 Contaminated Waste  
- 🧪 Infectious Waste  
- 🧴 Glass/Metal Recyclables  

Utilizing a **high-resolution camera**, the system captures real-time images and processes them using pretrained deep learning models. The model's output drives **servo motors** for automated waste sorting.

<img width="999" height="407" alt="Image" src="https://github.com/user-attachments/assets/bdef0ad0-911d-460e-a19a-ff4e1382c520" />

<img width="792" height="757" alt="Image" src="https://github.com/user-attachments/assets/dfb6c92a-52c3-435a-98fb-b83726ef6bce" />

![Image](https://github.com/user-attachments/assets/6f8687a9-9b22-41cc-a43a-1906370a8a0b)

![Image](https://github.com/user-attachments/assets/30be65e9-d39c-4da6-ae7a-d82eb32cfb45)

![Image](https://github.com/user-attachments/assets/763959f9-cbf4-41fa-81cc-f8a31618bb8d)

![Image](https://github.com/user-attachments/assets/aa31e1f1-ff2f-4e33-a4c2-66bd1034e5f4)

## 🧠 Technologies Used

- **CNN**: Custom convolutional neural network with 4.8M parameters  
- **Vision Transformer (ViT)**: Fine-tuned via HuggingFace Transformers  
- **YOLOv11**: Real-time object detection using Ultralytics  
- **EfficientDet-Lite0**: Lightweight model optimized for Raspberry Pi  
- **Raspberry Pi 4B**: For edge deployment  
- **TensorFlow, PyTorch, OpenCV, Pillow, Seaborn**  

## 📊 Dataset

- **Source**: Ubon Ratchathani University's "Pharmaceutical and Biomedical Waste" dataset  
- **Total Images**: 6,156  
- **Image Format**: RGB, 224x224 pixels  
- **Classes**: 13 (masks, gloves, syringes, tissues, etc.)  
- **Split**: 4300 training, 1056 validation, 800 testing  

## 🔧 Preprocessing

- Image resizing to 224x224  
- Pixel normalization [0, 1]  
- One-hot encoding of labels  
- Augmentation: flipping, rotation, zooming, brightness/contrast adjustment  

## 🏗️ Model Architectures

### 🔹 Vision Transformer (ViT)
- Patch-based image representation  
- Fine-tuned ViT-Base model using HuggingFace  

### 🔹 YOLOv11 (Ultralytics)
- YOLOv11n for real-time object detection  
- Trained on custom dataset with 640x640 input resolution  

### 🔹 EfficientDet-Lite0
- Compound-scaled, low-latency model  
- Exported to `.tflite` for edge deployment  

### 🔹 CNN
- Sequential conv layers with batch norm, dropout, and softmax  
- ~4.86M trainable parameters  

## 🧪 Results

| Model       | Accuracy | Precision | Recall | F1-Score |
|-------------|----------|-----------|--------|----------|
| CNN         | 93%      | 94%       | 94%    | 94%      |
| YOLOv11     | 93%      | 94%       | 94%    | 95%      |
| EfficientDet| 96%      | 95%       | 96%    | 95%      |
| **ViT**     | **98%**  | **96%**   | **97%**| **96%**  |

ViT consistently outperformed other architectures in both **classification accuracy** and **generalization** under noisy conditions.

## 🧰 Hardware & Deployment

- **Training**: macOS with Apple M3 chip, 16 GB RAM  
- **Deployment**: Raspberry Pi 4B with Pi Camera, MG90s servo motors  
- **Format**: `.tflite` and PyTorch `.pt` models  
- **Live Inference**: Real-time predictions via Pi Camera  

## 📈 Evaluation

- Accuracy, Precision, Recall, F1-score  
- Confusion matrix for per-class insights  
- Training vs Validation Loss plotted over epochs  
- Real-time video-based prediction visualizations  

## 🔮 Future Work

- Integration with **IoT** for cloud-based monitoring  
- Federated learning for **privacy-preserving** training  
- Expanding the dataset for better generalization  
- Deployment in **smart hospitals** and clinics  

## 🧑‍💻 Authors

- **Soubraylu Sivakumar** – [sivas.postbox@gmail.com](mailto:sivas.postbox@gmail.com)  
- **Abhay Kumar** – [ak8057@srmist.edu.in](mailto:ak8057@srmist.edu.in)  
- **Vishnu Gupta** – [vg0832@srmist.edu.in](mailto:vg0832@srmist.edu.in)

## 📚 References

[Refer to the paper for the full bibliography. Key works include research from IEEE Access, IEEE IoT Journal, and leading AI conferences.]

---

> **Note:** For full implementation, code, and pretrained models, refer to the `code/` and `models/` directories (or as available in the repo).

