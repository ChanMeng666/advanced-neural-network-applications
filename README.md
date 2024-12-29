<div align="center">
 <h1>Advanced Neural Network Applications</h1>

 <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
 <img src="https://img.shields.io/badge/Jupyter-F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white"/>
 <img src="https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
 <img src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white"/>
 <img src="https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white"/>
</div>

# 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Features](#-features)
- [Datasets](#-datasets)
- [Implementation Details](#-implementation-details)
- [Getting Started](#-getting-started)
- [Usage Examples](#-usage-examples)
- [Contributing](#-contributing)
- [License](#-license)

# 🔍 Project Overview
This repository demonstrates practical applications of neural network technologies through two distinct case studies:
1. Fish Classification using Perceptron Model
2. Heat Influx Prediction using Linear Neuron Model

The project showcases how neural networks can be applied to real-world classification and prediction problems, with detailed implementations and analysis in Python.

# ⚡ Features

### 🎯 Perceptron Implementation
- Custom perceptron model for binary classification
- Detailed weight update visualization
- Classification boundary analysis
- Performance metrics calculation

### 📈 Linear Neuron Model
- Single and multi-input implementations
- Batch learning demonstration
- 3D visualization of predictions
- Comprehensive error analysis

### 📊 Data Analysis Tools
- Custom data visualization functions
- Performance metric calculations
- Model comparison utilities
- Interactive Jupyter notebooks

# 📚 Datasets

### Fish Dataset
- Contains measurements of scale ring diameters
- Binary classification: Canadian vs Alaskan fish
- Features: freshwater and saltwater ring measurements
- Rich visualization of classification boundaries

### Heat Influx Dataset
- Records heat influx measurements from building elevations
- Features: North and South elevation measurements
- Target: Heat influx predictions
- Includes 3D visualization capabilities

# 🛠 Implementation Details

### Perceptron Model
```python
def perceptron(inputs, weights, bias):
    # Model implementation details
    activation = np.dot(inputs, weights) + bias
    return 1 if activation > 0 else 0
```

### Linear Neuron
```python
def linear_neuron(inputs, weights, bias):
    # Model implementation details
    return np.dot(inputs, weights) + bias
```

# 🚀 Getting Started

### Prerequisites
- Python 3.x
- Jupyter Notebook
- Required packages:
  ```bash
  pip install numpy pandas matplotlib scikit-learn
  ```

### Installation
1. Clone the repository
   ```bash
   git clone https://github.com/ChanMeng666/advanced-neural-network-applications.git
   ```
2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
3. Launch Jupyter Notebook
   ```bash
   jupyter notebook
   ```

# 💻 Usage Examples

### Fish Classification
```python
# Load and prepare data
fish_data = pd.read_csv('Fish_data.csv')

# Train perceptron model
model = train_perceptron(fish_data)

# Visualize results
plot_classification_boundary(model, fish_data)
```

### Heat Influx Prediction
```python
# Load and prepare data
heat_data = pd.read_csv('heat_influx_noth_south.csv')

# Train linear neuron
model = train_linear_neuron(heat_data)

# Visualize predictions
plot_3d_predictions(model, heat_data)
```

# 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

# 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

# 🔧 Tech Stack
![Python](https://img.shields.io/badge/python-%2314354C.svg?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-%23F37626.svg?style=for-the-badge&logo=Jupyter&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

## 🙋‍♀ Author

Created and maintained by [Chan Meng](https://github.com/ChanMeng666).
