[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/HTantjLn)
﻿# Project Title: Climate Change Impact Assessment and Prediction System for Nepal

### Project Goal: 
- Develop an end-to-end data analysis system that monitors, analyzes, and predicts climate change impacts in Nepal with a focus on vulnerable regions
### Target Audience: 
- Recent data science graduates applying their skills to real-world climate problems


## 1. Project Planning & Requirements Gathering

Analysis of climate change status of Nepal and developing prediction.

## 2. Data Collection & Acquisition

Weather & Climate Data:

- Historical temperature, precipitation, and extreme weather events data

## 3. Data Preprocessing & Storage

- Develop data cleaning pipelines for each data source
- Handle missing values through appropriate imputation techniques
- Normalize data from different sources and convert to consistent units
- Implement temporal alignment for time-series data
-- Design CSV schema for storing processed data
- Document data lineage and preprocessing steps
- Implement data validation checks for consistency

## 4. Exploratory Data Analysis (EDA)

- Analyze temperature trends across different regions and elevations
- Visualize precipitation patterns and changes over time
- Identify extreme weather event frequency and intensity changes
- Examine correlations between climate variables and environmental impacts
- Analyze glacial retreat rates over time
- Map climate vulnerability across different regions
- Create interactive visualizations of key trends
- Conduct statistical tests to validate observed changes

## 5. Feature Engineering

- Create derived climate indices (drought indices, heat stress metrics)
- Develop seasonal indicators for monsoon patterns
- Generate lag features for time-series prediction
- Create spatial proximity features for geospatial analysis
- Extract relevant features from satellite imagery
- Integrate geographic information into feature set
- Normalize and scale features appropriately
- Perform dimensionality reduction if needed

## 6. Machine Learning Model Development

### Classification Models:

- Random Forest for climate zone classification
- Support Vector Machines for extreme event prediction
- Gradient Boosting for vulnerability assessment
- Model evaluation using appropriate metrics (RMSE, MAE, F1-score)


### Regression Models:

- Multiple linear regression for impact assessment
- Ridge/Lasso regression for dealing with multicollinearity
- Gradient boosting regression for non-linear relationships
- Model validation through cross-validation techniques




## 8. Model Evaluation & Validation

- Establish appropriate evaluation metrics for different model types
- Implement cross-validation strategies for robust evaluation
- Conduct sensitivity analysis for key parameters
- Compare model performance against baseline approaches
- Evaluate model performance on different geographical regions
- Assess prediction accuracy for different time horizons
- Document uncertainty in model predictions
- Validate models against recent climate events

## 9. Dashboard Development with Streamlit

- Design user-friendly interface with multiple pages
- Create interactive maps showing climate vulnerability
- Implement time-series visualization components
- Develop model prediction interfaces
- Add filtering capabilities by region, time period, and climate variables
- Create downloadable report generation functionality
- Implement user feedback collection mechanism
- Ensure mobile-friendly design

## 10. Deployment & Integration

- Set up cloud-based hosting for the Streamlit application
- Configure automated data pipeline for regular updates
- Implement API endpoints for integration with other systems
- Set up continuous integration and deployment workflow
- Ensure appropriate security measures for sensitive data
- Document deployment architecture and dependencies
- Create system monitoring dashboard
- Implement backup and disaster recovery procedures

**This is developed for education purpose of castpone project of Omdena NIC Nepal. It is not recommended to use in realistic world**
streamlit app link - https://omdena-nic-nepal-capstone-project-prgiri422-app-xzlzyw.streamlit.app/
repository link - https://github.com/Omdena-NIC-Nepal/capstone-project-prgiri422.git
