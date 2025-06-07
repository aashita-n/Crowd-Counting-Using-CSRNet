# Crowd-Counting-Using-VCG16-and-CSRNet

This mini-project implements a deep learning-based crowd counting system designed to estimate the number of people in an image. It uses pretrained convolutional neural networks (CNNs), specifically **CSRNet** and **VGG16**, to generate density maps that reflect the spatial distribution of individuals in crowded scenes. The project was developed as part of our Deep Learning coursework.

---

## Project Overview

Crowd counting is a key task in fields like surveillance, event management, and public safety. This project leverages the power of **transfer learning** with pretrained models to perform accurate crowd estimation, especially in high-density environments where traditional methods fall short.

---

## Objective

- Utilize pretrained models (**CSRNet** and **VGG16**) for feature extraction.
- Predict density maps that represent crowd distribution in images.
- Estimate total crowd count by integrating over the density map.
- Visualize results and evaluate performance using real datasets.

---

## Dataset Used

- **ShanghaiTech Part A** (or equivalent sample dataset):
  - Includes images of highly crowded scenes.
  - Head annotations are converted to density maps using Gaussian kernels.

---

## Technologies and Libraries

- **Python**
- **TensorFlow / Keras**
- **CSRNet** (based on VGG16 frontend)
- **OpenCV**
- **NumPy / Pandas**
- **Matplotlib / Seaborn**
- **Scikit-learn**

---

## System Architecture 




---

## Steps Performed

1. **Data Preprocessing**
   - Resized and normalized crowd images.
   - Converted head annotations into ground truth density maps.

2. **Model Building**
   - Implemented **CSRNet**, which uses **VGG16** as its frontend.
   - Added dilated convolutions in the backend for preserving resolution.
   - Compiled with MSE loss and Adam optimizer.

3. **Training**
   - Trained on the processed dataset using pretrained weights.
   - Evaluated performance with MAE and MSE.
   - Plotted training vs. validation loss.

4. **Evaluation**
   - Compared predicted count with ground truth.
   - Visualized density maps overlaid on crowd images.
   - Assessed model robustness on unseen data.

---

## Results

- Achieved accurate crowd count predictions on dense scenes.
- CSRNet’s architecture helped preserve spatial details.
- Pretrained weights on VGG16 accelerated convergence and improved generalization.

---

## Future Enhancements

- Integrate real-time video input for dynamic crowd monitoring.
- Expand training with larger datasets (e.g., UCF-QNRF, WorldExpo).
- Experiment with alternate architectures like **MCNN** or **SaCNN**.
- Convert model for edge deployment using ONNX or TensorFlow Lite.

---

## Contributors

- **Aashita Narayanpur**  
- **Prajwal J.B**  

Mini Project submitted as part of the Deep Learning coursework  
Department of Artificial Intelligence & Data Science

---

## License

This project is for academic and learning purposes only.

---

> Note: The pretrained **weights file is included in the repository** for reproducibility and direct usage.

