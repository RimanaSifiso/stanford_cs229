# CS229: Machine Learning - Stanford
---
&copy; Sifiso Rimana, 2024
This repository serves as a storage for the notes and exercises as well as projects I make for CS229 Stanford's CS229 Machine Learning course. NB: While I did not officially enroll into the course, I followed it from Stanford Online YouTube channel. Below is the outline I use to follow the course (you'll notice that I use Python, PyTorch, Scikit-Learn, NumPy, and ```statsmodels```).


---

### **1. Introduction and Pre-requisites Review (Weeks 1–2)**

#### **Topics**
- Review:
  - Linear Algebra (vectors, matrices, eigenvalues).
  - Matrix Calculus (gradients, Jacobians).
  - Probability Theory and Statistics.

#### **Practicals**
- Use NumPy/`torch` for:
  - Vector and matrix operations (e.g., dot products, eigenvalues).
  - Differentiating simple functions (e.g., gradients of quadratic forms).
  - Sampling from distributions (`torch.distributions`).
- Simple probability visualizations (e.g., normal distribution curve).

---

### **2. Supervised Learning (Weeks 3–6)**

#### **Topics**
- **Linear Regression**
  - Gradient Descent, Normal Equations, MLE.
- **Logistic Regression**
  - Sigmoid function, Newton's Method.
- **Generalized Linear Models (GLM)** and Exponential Family.
- **Support Vector Machines (SVM)** and Kernel Methods.

#### **Practicals**
- **scikit-learn**:
  - Linear regression: fit a model to predict house prices.
  - Logistic regression: classify binary classes (e.g., Iris dataset).
  - SVM: Use `SVC` for non-linear decision boundaries.
- **PyTorch**:
  - Implement linear and logistic regression models from scratch using `torch.nn`.
  - Train with `torch.optim.SGD` and visualize loss curves.
  - Implement SVM using PyTorch.

---

### **3. Neural Networks and Deep Learning (Weeks 7–8)**

#### **Topics**
- **Basics of Neural Networks**:
  - Perceptron, feedforward networks, backpropagation.
  - Deep Learning introduction.
- Optimization techniques (SGD, Adam).

#### **Practicals**
- **PyTorch**:
  - Build a simple neural network for MNIST digit classification.
  - Train the network using `torch.nn.CrossEntropyLoss`.
  - Experiment with optimizers (SGD vs Adam).
- Compare performance with scikit-learn's MLPClassifier.

---

### **4. Statistical Learning Theory (Weeks 9–10)**

#### **Topics**
- Bias-Variance Tradeoff.
- Regularization (Ridge, Lasso, Dropout).
- Model Selection.

#### **Practicals**
- **scikit-learn**:
  - Apply Lasso and Ridge regression to overfitting problems.
  - Use cross-validation for model selection.
- **PyTorch**:
  - Implement dropout in a neural network.
  - Visualize underfitting vs. overfitting.

---

### **5. Reinforcement Learning (Weeks 11–12)**

#### **Topics**
- Markov Decision Processes (MDP).
- Value Iteration, Policy Iteration.
- Basics of Q-Learning.

#### **Practicals**
- Implement basic Q-learning using NumPy.
- Explore reinforcement learning libraries like `Stable-Baselines3` or PyTorch.

---

### **6. Unsupervised Learning (Weeks 13–14)**

#### **Topics**
- K-Means Clustering.
- Gaussian Mixture Models (GMM).
- Principal Components Analysis (PCA).
- Independent Components Analysis (ICA).

#### **Practicals**
- **scikit-learn**:
  - K-Means on Iris or custom datasets.
  - PCA for dimensionality reduction (visualize results on 2D plots).
  - Apply GMM for clustering tasks.
- **PyTorch**:
  - Implement PCA using Singular Value Decomposition (SVD).

---

### **7. Variational Inference and VAEs (Week 15)**

#### **Topics**
- Variational Autoencoder (VAE).
- EM Algorithm.

#### **Practicals**
- **PyTorch**:
  - Implement a simple VAE for MNIST data.
  - Visualize latent space by plotting encoded representations.

---

### **8. Evaluation Metrics and Wrap-Up (Week 16)**

#### **Topics**
- Metrics (accuracy, precision, recall, F1-score).
- KL-Divergence, calibration.

#### **Practicals**
- **scikit-learn**:
  - Evaluate classification models using various metrics.
- **PyTorch**:
  - Calculate metrics during training and testing.

---

### Weekly Practical Setup
Each week can have:
1. **Theory:** Follow Stanford's lecture notes/slides.
2. **Practical:** Implement ideas with **scikit-learn** and **PyTorch**.
3. **Assignments:** Solve a real-world task (e.g., regression, classification, or clustering).

Would you like me to design a specific week in more detail?
