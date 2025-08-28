# Advanced ML Analytics Platform

![ML Analytics Platform](https://img.shields.io/badge/ML-Analytics%20Platform-blue)
![Python](https://img.shields.io/badge/Python-3.9%2B-brightgreen)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

A comprehensive machine learning analytics platform built with Streamlit that enables data exploration, preprocessing, model training, evaluation, and prediction—all through an intuitive web interface. This tool democratizes machine learning by allowing users without extensive programming knowledge to build and evaluate predictive models.

![Application Demo](https://modelbattle.streamlit.app/)

## 🚀 Features

### 📊 Data Acquisition & Exploration
- Upload CSV or JSON datasets or choose from built-in examples
- Comprehensive exploratory data analysis with interactive visualizations
- Detailed statistics, correlations, and distributions with interpretations
- Missing value analysis and outlier detection

### 🔧 Data Preprocessing
- Intelligent handling of missing values with multiple imputation methods
- Automated categorical encoding (Label Encoding, One-Hot Encoding)
- Feature scaling with visualization of results
- Advanced dimensionality reduction with PCA

### 🤖 Machine Learning Pipeline
- Automated task detection (Classification vs Regression)
- Multiple model selection with comparative training
- Hyperparameter tuning with GridSearchCV
- Feature importance visualization and interpretation

### 📈 Model Evaluation
- Comprehensive metrics for classification and regression tasks
- Interactive confusion matrix analysis
- ROC & PR curves for classification models
- Residual analysis for regression models

### 🔮 Prediction System
- Interactive prediction interface with smart defaults
- Quick prediction examples (average, high, low values)
- Z-score warnings for unusual input values
- Confidence intervals for predictions (where applicable)
- Feature contribution analysis for interpretability

## 📋 Requirements

```
streamlit>=1.25.0
pandas>=1.3.0
numpy>=1.20.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
statsmodels>=0.13.0
requests>=2.27.0
```

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/rayrohit999/advanced-ml-analytics-platform.git
cd advanced-ml-analytics-platform
```

2. Create and activate a virtual environment:
```bash
# On Windows
python -m venv venv
.\venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run app.py
```

5. Access the application in your web browser:
```
http://localhost:8501
```

## 💡 Usage Guide

### Data Upload
- Upload your CSV or JSON dataset using the file uploader
- Alternatively, select from example datasets or use built-in scikit-learn datasets

### Exploratory Data Analysis
- Explore summary statistics, distributions, and correlations
- Analyze missing values and detect outliers
- Visualize relationships between features

### Data Preparation
- Select your target variable for prediction
- Configure train-test split parameters
- Apply preprocessing techniques like imputation and encoding

### Model Selection & Training
- Choose models appropriate for your task (classification or regression)
- Optionally enable hyperparameter tuning
- Train and compare multiple models

### Evaluation & Insights
- Review model performance metrics
- Analyze confusion matrices or residual plots
- Examine feature importance

### Make Predictions
- Enter custom values or use quick prediction examples
- Get predictions with confidence intervals
- Understand feature contributions to predictions

## 🧪 Example Datasets

The platform includes several example datasets:
- **Iris Classification**: Flower classification based on measurements
- **Titanic Survival**: Predicting passenger survival on the Titanic
- **Wine Quality**: Predicting wine quality based on chemical properties
- **Countries Data**: Various statistics for countries worldwide

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📃 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact

Rohit - [@rayrohit999](https://github.com/rayrohit999)

Project Link: [https://github.com/rayrohit999/advanced-ml-analytics-platform](https://github.com/rayrohit999/advanced-ml-analytics-platform)

## 🙏 Acknowledgements

- [Streamlit](https://streamlit.io/)
- [Scikit-learn](https://scikit-learn.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
