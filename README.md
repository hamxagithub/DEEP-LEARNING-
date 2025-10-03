# Deep Feedforward Neural Network (DFNN) from Scratch with Visualization

## 📌 Assignment Overview
This project implements a **Deep Feedforward Neural Network (DFNN)** entirely from scratch using **Python + NumPy** for a **binary classification task**.  
It includes forward & backward propagation, training with gradient descent, visualization of the **decision boundary**, and exploration of **regularization techniques**.

---

## 🎯 Objectives
- Implement core DFNN mechanics (forward & backward passes).
- Train a binary classifier using synthetic datasets.
- Visualize **training loss curves** and **decision boundaries**.
- Experiment with **regularization, dropout, batch normalization, and K-Fold CV**.
- Gain deeper insights into **overfitting and generalization**.

---

## 🧩 Part A – Core DFNN Implementation

### 1. Activation Functions
- **Sigmoid**  
  σ(z) = 1 / (1 + e^(-z))  
  Derivative: σ′(z) = σ(z)(1 − σ(z))

- **ReLU**  
  ReLU(z) = max(0, z)  
  Derivative: ReLU′(z) = {1 if z>0, 0 otherwise}

- **Tanh**  
  tanh(z) = (e^z − e^(-z)) / (e^z + e^(-z))  
  Derivative: 1 − (tanh(z))²

### 2. Loss Function
- **Binary Cross-Entropy Loss**  
  L = - 1/N Σ [ y log(ŷ) + (1−y) log(1−ŷ) ]

### 3. `DeepNeuralNetwork` Class
Implements:
- `__init__`: Initialize weights & biases
- `forward_pass`: Propagation through layers
- `backward_pass`: Backpropagation with gradients
- `update_parameters`: Gradient descent
- `train`: Training loop with loss tracking
- `predict`: Generate binary predictions

---

## 🧩 Part B – Data & Visualization

- **Dataset**: Generated using `sklearn.datasets.make_moons()` or `make_circles()`.
- **Visualizations**:
  - Scatter plot of dataset
  - Training loss over epochs
  - Learned decision boundary

---

## 🧩 Part C – Regularization & Experiments

### 1. Regularization
- Drop irrelevant features (`Name`, `Ticket`, `Cabin`)  
- Compile with Adam optimizer  
- Key questions:
  - Why do we need activation functions?
  - What happens if we remove them?

### 2. Overfitting Analysis
- Train for **EPOCHS = 30**
- Compare **training vs validation curves**
- Identify **overfitting signs**

### 3. Regularization Techniques
- Implement assigned technique (Dropout, L2, BatchNorm, etc.)
- Evaluate:
  - Did training accuracy decrease?
  - Did validation accuracy improve?

### 4. K-Fold Cross Validation
- Run **5-fold CV**
- Compare **average accuracy vs single split**
- Discussion:
  - Why is K-Fold better for small datasets?
  - Tradeoff with higher K values?

---

## 🧩 Part D – Coding Tasks

### Q6. Dataset Augmentation
- Use `make_moons()`
- Apply Gaussian noise / flipping
- Plot **original vs augmented dataset**
- Explain how augmentation reduces overfitting

### Q7. Dropout Experiment
- Build 2-hidden-layer MLP (PyTorch/Keras)
- Train with **dropout (p=0.5)** and **without dropout**
- Plot training & validation losses
- Discuss generalization

### Q8. Batch Normalization
- Add BatchNorm after hidden layers
- Compare convergence speed
- Explain stabilization effects

### Q9. Hyperparameter Tuning with K-Fold
- Try architectures: `[16,8,1]`, `[32,16,1]`, `[64,32,1]`
- Use 5-fold CV
- Print **mean validation accuracy** & best model

### Q10. L2 Regularization
- Train with and without **L2 penalty**
- Compare training vs validation losses
- Show weight norms & impact on overfitting

---

## 📊 Expected Outputs
- Plots:
  - Training loss curve
  - Dataset scatter plot
  - Decision boundary
  - Regularization comparisons
- Accuracy metrics:
  - Training accuracy
  - Cross-validation accuracy
- Written discussion for reflection questions

---

## 🚀 How to Run
1. Clone repo & install dependencies:
   ```bash
   pip install numpy matplotlib scikit-learn
