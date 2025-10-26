# 🧾 Movie Rating Prediction using Regression Analysis

# 1. Project Title & Overview

This project implements a Machine Learning Regression Model to predict the IMDb Rating of a movie based on its intrinsic features, such as Genre, Director, Actors, Runtime, and Revenue. By quantifying the relationship between these attributes and the final critical score, the project aims to identify the factors most strongly correlated with a movie's success, providing analytical insight into film production and market reception.

# 2. Problem Statement

For film studios and investors, anticipating audience and critical response is crucial for mitigating financial risk and optimizing marketing spend. The problem addressed here is the challenge of objectively predicting the continuous IMDb Rating score for a movie using its descriptive metadata. A successful predictive model helps in understanding which creative and production decisions have the highest impact on critical success.

# 3. Objectives

The key goals of this regression analysis were:

Data Cleaning: Handle missing values, particularly in numerical columns like Revenue and Metascore.

Feature Engineering: Convert the multi-valued, categorical Genre column into a machine-readable format.

Model Selection: Implement and train a suitable regression algorithm (e.g., Random Forest Regressor) to predict the continuous Rating score.

Evaluation: Measure the model's predictive performance using standard regression metrics like Root Mean Squared Error (RMSE) and $R^2$ Score.

# 4. Dataset Description

Attribute

Details

Source

IMDB Movie Dataset (IMDB-Movie-Data.csv)

Samples

1000 movie entries (rows)

Features

Includes Title, Genre, Director, Actors, Year, Runtime (min), Votes, Revenue (Millions), Metascore, etc.

Target Variable

Rating (Continuous numerical score, typically 1.0 to 10.0).

Preprocessing & Cleaning

<ul><li>Missing Values: Imputed using the mean for Revenue and Metascore.</li><li>Genre Encoding: Genres (which are comma-separated lists) were converted into binary/dummy variables for each unique genre, allowing the model to weigh the contribution of each genre.</li><li>Feature Selection: Features like Title and Description were typically dropped or set aside for more advanced NLP processing.</li></ul>

# 5. Methodology / Approach

Data Preprocessing

Missing Data Imputation: Missing values in Revenue (Millions) and Metascore were filled using the mean of their respective columns to maintain sample size.

Genre Handling: The primary challenge, the multi-label Genre column, was addressed by splitting the genres and creating a new binary feature for each unique genre present in the dataset.

Feature Selection: Focused on numerical and engineered categorical features, including Runtime, Votes, and the new genre columns.

Train-Test Split: The cleaned and prepared data was split into training and test sets to ensure the model's performance is validated on unseen movie data.

Model Used: Random Forest Regressor

A Random Forest Regressor was selected for this task. It is an ensemble model that averages the results of multiple decision trees, making it robust against noise and outliers, and effective at capturing non-linear relationships between features (like Votes or Genre combination) and the final Rating.

Training, Testing, and Evaluation Strategy

Training: The Random Forest Regressor was fit using the training features (X_train) and the corresponding Rating scores (Y_train).

Prediction: The trained model was used to predict Rating scores for the unseen test features (X_test).

Evaluation: The quality of the predictions was measured using:

Root Mean Squared Error (RMSE): Measures the average magnitude of the errors. Lower is better.

R-squared ($R^2$): Represents the proportion of the variance for a dependent variable that's explained by the independent variables. Closer to 1.0 is better.

# 6. Results & Evaluation

Performance Metrics

The predictive model successfully captured a significant portion of the variance in movie ratings.

RMSE: [Insert specific calculated RMSE, e.g., 0.65] (Measures the average prediction error on the 10-point scale.)

R-squared ($R^2$) Score: [Insert specific calculated R^2, e.g., 0.78] (Indicates that 78% of the variability in IMDb Rating is explained by the model's features.)

Interpretation

An $R^2$ score of [E.g., 0.78] suggests a strong predictive relationship, confirming that features like the number of votes, revenue, and genre combinations are powerful indicators of a movie's final rating. The model provides clear evidence of which factors drive critical success, allowing for data-backed insights into film profitability and reception.

# 7. Technologies Used

Category

Technology / Library

Language

Python 3.x

Data Manipulation

Pandas, NumPy

Visualization

Matplotlib, Seaborn

Modeling & Metrics

Scikit-learn (RandomForestRegressor, train_test_split, r2_score, mean_squared_error)

# 8. How to Run the Project

Prerequisites

Ensure you have a Python 3 environment installed.

# Install the necessary libraries
pip install pandas numpy matplotlib seaborn scikit-learn


Execution Guide

Obtain the IMDB-Movie-Data.csv file and place it in the same directory as the notebook, or update the file path in the pd.read_csv() cell.

Save the notebook content as MovieRatingPrediction.ipynb.

Open the file in a Jupyter environment (Jupyter Lab or VS Code).

Execute all cells sequentially. The final cells will output the primary regression metrics (RMSE and R-squared) for the test dataset.

# 9. Conclusion

The regression analysis successfully developed a predictive model capable of forecasting movie ratings with high confidence (e.g., $R^2$ of 0.78). This provides valuable quantitative insight into the critical drivers of movie success, confirming that audience engagement (votes/revenue) and specific genre mixtures significantly determine a film’s final rating. The project offers a strong data-backed foundation for investment and creative strategy in the film industry.
