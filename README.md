# DATA-PIPELINE-DEVELOPMENT

*COMPANY* : CODETECH IT SOLUTIONS

*NAME* : HRIDAYI HEMANT BHELLI

*INTERN ID* : CTIS8232

*DOMAIN* : DATA SCIENCE

*DURATION* : 4 WEEKS

*MENTOR* : NEELA SANTOSH

Project Title: Data Pipeline Development for Data Preprocessing, Transformation, and Machine Learning

This project focuses on developing an automated data pipeline for preprocessing, transforming, and training machine learning models using pandas and scikit-learn. The main objective of the project is to build a structured and reusable workflow that can take raw data, clean it, transform it into a machine-readable format, train a machine learning model, and save the complete pipeline for future use. This project demonstrates how data engineering and machine learning can work together to create efficient predictive systems.

The project was implemented using Python because it is widely used in data science and machine learning due to its powerful libraries and ease of development. The code was written and executed using development platforms such as Visual Studio Code along with Windows PowerShell for package installation and script execution. Required libraries such as pandas, scikit-learn, and joblib were installed using Python package management tools.

For the dataset, the project used the Titanic Dataset from Kaggle. This dataset contains passenger-related information such as age, gender, ticket class, fare amount, and survival status. The target variable for prediction was whether a passenger survived or not. This dataset was chosen because it contains both numerical and categorical data, as well as missing values, making it suitable for testing preprocessing techniques.

The project begins by loading the dataset using pandas and separating the target variable from the input features. After loading the data, the pipeline automatically identifies numerical and categorical columns. This makes the system flexible enough to work with similar structured datasets.

The preprocessing stage was built using scikit-learn’s Pipeline and ColumnTransformer modules. For numerical data, missing values were handled using mean imputation, and features were standardized using scaling techniques. For categorical data, missing values were replaced using the most frequent category, and categorical values were converted into numerical format using one-hot encoding. These preprocessing steps ensure that the data becomes suitable for machine learning algorithms.

After preprocessing, a Random Forest classification model was added to the pipeline. The dataset was divided into training and testing sets to evaluate model performance on unseen data. The model was trained using the training set and later tested using evaluation metrics such as accuracy and classification reports.

A major feature of this project is pipeline persistence. Using joblib, the complete trained pipeline was saved as a file named data_pipeline.pkl. This saved file includes both preprocessing steps and the trained model, allowing the system to make predictions on new data without retraining.

This project has practical applications in healthcare, finance, e-commerce, fraud detection, customer analytics, and predictive business systems. It provides strong foundational knowledge in data preprocessing, feature engineering, machine learning workflow design, and model deployment preparation.
