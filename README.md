### Languages and Tools

- **Python**: For data preprocessing, model training, and evaluation.
- **Pandas**: For data manipulation and preparation.
- **Scikit-learn**: For implementing linear and polynomial regression models.
- **Jupyter Notebook**: For developing and running the analysis.
- **Matplotlib/Seaborn**: For data visualization.

# Life Expectancy Prediction using Linear and Polynomial Regression Models

## Overview

This project focuses on predicting life expectancy using various linear and polynomial regression models. The aim is to develop a machine learning model that predicts a person’s life expectancy based on features related to the region they were born in, such as environmental factors, socioeconomic status, and healthcare indicators. The project incorporates data preprocessing, model evaluation, and hyperparameter tuning to improve the performance and generalizability of the model.

### Key Features

1. **Data Preprocessing**:
   - The dataset contains approximately 2,000 instances with 20 features related to life expectancy.
   - Preprocessing techniques include handling outliers, normal transformations, feature scaling, and splitting the data into training, validation, and test sets.
   - The preprocessing pipeline ensures that new data can be processed in a consistent and automated manner.

2. **Exploratory Data Analysis (EDA)**:
   - EDA was performed to understand the data distributions, identify outliers, and assess relationships between the features and the target variable (life expectancy).
   - Visualizations such as correlation matrices, linearity graphs, and outlier percentage plots were generated to explore the data.

3. **Regression Models**:
   - The project experimented with various regression models:
     - **Linear Regression**: Baseline model that demonstrated underfitting, with reasonable performance but missed some underlying patterns in the data.
     - **Polynomial Regression**: A third-degree polynomial model was initially tested but overfit the training data. A second-degree polynomial model with regularization was selected as the optimal model.
     - **Regularization (Lasso)**: Lasso regression was applied to reduce overfitting by penalizing large coefficients and improving model generalization.

4. **Model Evaluation**:
   - **Cross-validation** was used to evaluate model performance more robustly.
   - Key metrics include **Root Mean Squared Error (RMSE)** and **R-squared (R²)**, providing insights into how well the models fit the data.

5. **Hyperparameter Tuning**:
   - Grid search was used to optimize the hyperparameters, such as Lasso's regularization strength (alpha), scaling methods, and transformations.
   - The best-performing model was a second-degree polynomial with Lasso regularization, achieving an R² score of 0.74 and an RMSE of 4.3 on the validation set.

### System Design and Methodology

1. **Data Splitting**:
   - The dataset was split into training (50%), validation (20%), and test (30%) sets. Careful attention was given to ensure consistent distributions across these sets to avoid overfitting and underfitting.

2. **Model Training**:
   - The models were trained using various regression techniques, and their performance was evaluated on the validation set. Regularization techniques were applied to avoid overfitting.

3. **Model Selection**:
   - After experimenting with different models, the second-degree polynomial regression with Lasso regularization was chosen as the final model. It struck a balance between complexity and performance, avoiding the extremes of underfitting and overfitting.

### Files in the Repository

- **`code.ipynb`**: Jupyter Notebook containing the Python code for data preprocessing, model training, evaluation, and hyperparameter tuning.
- **`predictions.csv`**: Final predictions on the test set based on the chosen model.
- **`report.pdf`**: Detailed report explaining the methodology, model selection process, and results.

### How to Run the Project

1. **Set Up Jupyter Notebook**:
   - Install Jupyter Notebook using Anaconda or any preferred method.
   - Open the `code.ipynb` file and run the cells to preprocess the data, train the models, and evaluate the results.

2. **Run the Notebook**:
   - The notebook contains all the steps necessary for data preprocessing, training the models, and performing predictions.
   
3. **View Predictions**:
   - The final predictions are saved in `predictions.csv`, which can be reviewed after running the notebook.

4. **Report**:
   - Open `report.pdf` to view the detailed methodology and results.

### Data Source

The dataset is related to life expectancy and was provided as part of a machine learning assignment. It includes features such as environmental, healthcare, and socioeconomic factors, all of which influence life expectancy.
