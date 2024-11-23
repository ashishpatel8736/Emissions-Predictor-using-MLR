# 🚗 Emissions Predictor Using Multiple Linear Regression (MLR)

**Emissions Predictor Using MLR** is a machine learning application that uses a **Multiple Linear Regression (MLR)** model to predict carbon dioxide emissions (g/km) of a vehicle based on multiple features like engine size, fuel consumption, and other vehicle characteristics. This project demonstrates how to build, train, evaluate, and deploy a regression model using Python and Streamlit.

![Emissions Predictor Banner](banner_md.jpeg)

---

## 🌟 Features
- **Multiple Linear Regression (MLR) Model**: Accurate prediction of CO2 emissions using multiple features.
- **Interactive User Interface**: Built using Streamlit for an easy-to-use experience.
- **Real-Time Predictions**: Adjust features like engine size and fuel consumption to get instant CO2 emission predictions.
- **Data Visualization**: Displays visualizations to understand the relationships between vehicle features and CO2 emissions.

---

## 🛠️ Tech Stack
- **Python**: Core programming language.
- **Streamlit**: Framework for creating the interactive web application.
- **Scikit-learn**: Machine learning library used for training the MLR model.
- **Matplotlib**: For data visualization.
- **Pandas**: For data manipulation and analysis.

---

## 🚀 How It Works
1. **Input Features**: Use the sliders to select features like engine size and fuel consumption.
2. **Real-Time Prediction**: The app instantly predicts the CO2 emissions based on the input.
3. **Visualize Data**: See scatterplots and other visualizations to understand the relationship between features and emissions.

---
## 📂 Repository Structure
```plaintext

📦 Emissions-Predictor-using-MLR

├──  app.py 
├──  mlr_model.pkl 
├──  ridge_tuned_model.pkl
├──  scaler.pkl
├──  CO2 Emissions Predictor using MLR.ipynb
├── README.md                 
├── vehicle_data.csv 
├── requirements.txt           
├── LICENSE                   
├── banner_md.jpeg             
```

---

## 🖥️ Installation and Usage

### Prerequisites
- Python 3.8 or higher installed on your machine.

### Setup Instructions

### Step 1: Clone the Repository
```bash
git clone https://github.com/ashishpatel8736/Emissions-Predictor-using-MLR.git
   cd CO2-Emissions-Predictor
```

### Step 2: Install Dependencies
Ensure you have Python installed. Run the following to install the required libraries:
```bash
pip install -r requirements.txt

```

### Step 3: Start the Application
Run the Streamlit app:
```bash
streamlit run app.py
```

### Step 4: Open your browser and go to:

```bash
http://localhost:8501

```


## 📊 Sample Data
Here is an example of the dataset used for training the SLR model:

| Engine Size (L) | Fuel Consumption (L/100km)| CO2 Emissions (g/km) |
|------------------|----------------------|----------------------|
| 1.5              | 6.5                  | 145                  |
| 2.0              | 7.0                  | 185                  |
| 3.0              | 8.5                  | 250                  |
| 4.0              | 9.0                  | 320                  |
| 5.0              | 10.5                 | 400                  |


---

## 🎯 Future Enhancements
- Add support for more input features, such as vehicle weight or fuel type.
- Implement model optimization techniques like cross-validation for better accuracy.
- Add support for uploading custom datasets.
- Provide downloadable results and summary reports.




---


## 🤝 Contributing
Contributions are welcome! If you'd like to contribute, please:

1. **Fork the repository**.
2. **Create a feature branch**.
3. **Submit a pull request**.



## 🙌 Acknowledgements
- **Scikit-learn** for providing robust machine learning tools.
- **Streamlit** for enabling easy deployment of ML apps.
- **Pandas and Matplotlib** for data manipulation and visualization.

---
## 🛡️ License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author
**Your Name**  
[GitHub](https://github.com/ashishpatel8736) | [LinkedIn](https://www.linkedin.com/in/ashishpatel8736)


