# Regression Trees

**Predicting Weight Using Height and Gender in Regression Tree Models**

---

## Introduction

This project focuses on predicting individuals' weights based on their heights and genders. By analyzing these relationships, we can derive valuable insights into health and wellness. The study compares two modeling approaches: **linear regression** and **regression tree models**.

---

## Objective

The primary goal of the project is to evaluate how well weight can be predicted using height and gender as inputs. Additionally, the project investigates how adding gender as a variable affects model performance.

---

## Workflow

The project follows these steps:

1. Import the dataset containing height, weight, and gender data.
2. Explore and visualize the data to understand its distribution and relationships.
3. Develop a **linear regression model** to predict weight using height as the independent variable.
4. Evaluate the performance of the linear regression model using error metrics.
5. Incorporate gender into the model to examine its impact on predictions.
6. Build a **regression tree model** to predict weight using both height and gender.
7. Compare the performance of the regression tree to the linear regression model.

---

## Tools and Libraries

The project employs several libraries and tools, including:

- **Data Manipulation and Visualization**:
  - Libraries like `tidyverse` for data cleaning and visualization.
- **Statistical Modeling**:
  - `broom` for converting statistical models into tidy data frames.
  - `finalfit` for creating comprehensive statistical summaries.
- **Model Building**:
  - `rpart` for building regression trees.
  - `rpart.plot` for visualizing decision trees.
- **Evaluation Metrics**:
  - `Metrics` for calculating key performance indicators such as mean error and RMSE.

---

## Key Findings

1. **Linear Regression**: Height alone is a useful predictor of weight, with a linear relationship evident in the data.
2. **Regression Tree Models**: Incorporating gender significantly improves the model’s accuracy, as it accounts for variations in weight across genders.
3. **Performance Evaluation**: Error metrics and visualization tools provide a comprehensive understanding of model accuracy and predictive power.

---

## Conclusion

This project demonstrates the effectiveness of regression tree models in predicting weight using height and gender. It highlights the importance of considering multiple variables for more accurate predictions. These insights can be applied in various domains, including health and wellness, where predicting anthropometric measures is critical.

---

## Future Work

Further enhancements could include:

- Adding more variables such as age, lifestyle habits, or ethnicity to improve the model.
- Exploring advanced machine learning models like random forests or gradient boosting for better accuracy.
- Applying the models to other datasets to test generalizability.

---

**Author**: Nyakundi  
**Contact**: nyakundicaleb98@gmail.com
