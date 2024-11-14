# [SQL Project] HR Analysis: Employee Attrition Analysis

## Business Problem
One of the challenges for companies is employee attrition, particularly when it leads to the loss of skilled workers, increased recruitment costs, and reduced overall productivity. Understanding the employee attrition patterns and identifying key factors that influence attrition can help companies develop strategies to improve employee retention.

## Business Questions
1. What is the overall attrition rate in the company?
2. Which job roles have the highest attrition rates?
3. What factors are associated with high attrition rates in job roles with the highest attrition rates?
4. Are there spesific employee groups that are more likely leave the company?
5. How does attrition trend vary across different levels of total working experience?

## What's in this project?
1. Data cleansing:
   - Identify and handling missing values
   - Data correction: changee data type, fix incorrect values in categorical columns
   - Identify duplication
   - Outlier detection and removal
   - Dropping unnecessary column
   - Label encoding
2. Exploratory Data Analysis (EDA)

## Summary
1. Overall attrition rate in this company is 17.1%
2. Sales Representatives have the highest attrition rate (39.76%), followed by Laboratory Technicians (24.03%) and Human Resources (23.08%)
3. Key factors that associated with high attrition rates in 3 job roles with the highest attrition rates are job satisfaction, working over time, and monthly income
4. Groups that are more likely to leave the company include employees who are single, those in the Human Resources department, male employees, those earning lower monthly income, employees who feel dissatisfaction with their job, those who have a longer distance from home
5. There is a different trend in working years for each age range.
   - Negative trend of attrition rates for employees with 0-10 years range but it still higher than 11-30 range
   - Relatively stable at 11-30 years range
   - Significant peak for employees with 31-50 years experience.

## Recommendation
1. Improve employee retention, such as identify and resolve job disssatisfaction (example by survey employees) or reviewing overtime policies.
2. Consider transportation support to reduce the impact of long commutes on employee retention.
3. Make different approach for each working years range:
    - 0-10 years: focus on improving or providing growth opportunities, such as career development.
    - 11-30 years: maintain engagement by rewarding long-term contributions
     -31-40 years: implementing targeted retirement planning or flexible work options.