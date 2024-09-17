# Import Dataset


```python
import pandas as pd
import numpy as np
import seaborn as sns
```


```python
df = pd.read_csv('D:/Skills/Portfolio/Dataset/HR-Employee-Attrition-Messy.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Attrition</th>
      <th>BusinessTravel</th>
      <th>DailyRate</th>
      <th>Department</th>
      <th>DistanceFromHome</th>
      <th>Education</th>
      <th>EducationField</th>
      <th>EmployeeCount</th>
      <th>EmployeeNumber</th>
      <th>...</th>
      <th>RelationshipSatisfaction</th>
      <th>StandardHours</th>
      <th>StockOptionLevel</th>
      <th>TotalWorkingYears</th>
      <th>TrainingTimesLastYear</th>
      <th>WorkLifeBalance</th>
      <th>YearsAtCompany</th>
      <th>YearsInCurrentRole</th>
      <th>YearsSinceLastPromotion</th>
      <th>YearsWithCurrManager</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>41.0</td>
      <td>Yes</td>
      <td>Travel_Rarely</td>
      <td>1102</td>
      <td>Sales</td>
      <td>1</td>
      <td>2</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>80</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>49.0</td>
      <td>No</td>
      <td>Travel_Frequently</td>
      <td>279</td>
      <td>Research &amp; Development</td>
      <td>8</td>
      <td>1</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>2</td>
      <td>...</td>
      <td>4</td>
      <td>80</td>
      <td>1</td>
      <td>10</td>
      <td>3</td>
      <td>3</td>
      <td>10</td>
      <td>7</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>37.0</td>
      <td>Yes</td>
      <td>Travel_Rarely</td>
      <td>1373</td>
      <td>Research &amp; Development</td>
      <td>2</td>
      <td>2</td>
      <td>Other</td>
      <td>1</td>
      <td>4</td>
      <td>...</td>
      <td>2</td>
      <td>80</td>
      <td>0</td>
      <td>7</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33.0</td>
      <td>No</td>
      <td>Travel_Frequently</td>
      <td>1392</td>
      <td>Research &amp; Development</td>
      <td>3</td>
      <td>4</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>5</td>
      <td>...</td>
      <td>3</td>
      <td>80</td>
      <td>0</td>
      <td>8</td>
      <td>3</td>
      <td>3</td>
      <td>8</td>
      <td>7</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>27.0</td>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>591</td>
      <td>Research &amp; Development</td>
      <td>2</td>
      <td>1</td>
      <td>Medical</td>
      <td>1</td>
      <td>7</td>
      <td>...</td>
      <td>4</td>
      <td>80</td>
      <td>1</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 35 columns</p>
</div>



# Data Cleansing

## Missing Values


```python
#check missing values
missing_values = df.isnull().sum().reset_index()
missing_values.columns = ['Column', 'Total Missing Values']

#show columns that have missing values
missing_values_count = missing_values[missing_values['Total Missing Values'] > 0]
missing_values_count
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column</th>
      <th>Total Missing Values</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Age</td>
      <td>15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Department</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_clean = df.copy()

#imputation 'Age' with mean
df_clean['Age'] = df_clean['Age'].fillna(df_clean['Age'].mean().round())

#imputation 'Department' with mode
df_clean['Department'] = df_clean['Department'].fillna(df_clean['Department'].mode()[0])

df_clean
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Attrition</th>
      <th>BusinessTravel</th>
      <th>DailyRate</th>
      <th>Department</th>
      <th>DistanceFromHome</th>
      <th>Education</th>
      <th>EducationField</th>
      <th>EmployeeCount</th>
      <th>EmployeeNumber</th>
      <th>...</th>
      <th>RelationshipSatisfaction</th>
      <th>StandardHours</th>
      <th>StockOptionLevel</th>
      <th>TotalWorkingYears</th>
      <th>TrainingTimesLastYear</th>
      <th>WorkLifeBalance</th>
      <th>YearsAtCompany</th>
      <th>YearsInCurrentRole</th>
      <th>YearsSinceLastPromotion</th>
      <th>YearsWithCurrManager</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>41.0</td>
      <td>Yes</td>
      <td>Travel_Rarely</td>
      <td>1102</td>
      <td>Sales</td>
      <td>1</td>
      <td>2</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>80</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>49.0</td>
      <td>No</td>
      <td>Travel_Frequently</td>
      <td>279</td>
      <td>Research &amp; Development</td>
      <td>8</td>
      <td>1</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>2</td>
      <td>...</td>
      <td>4</td>
      <td>80</td>
      <td>1</td>
      <td>10</td>
      <td>3</td>
      <td>3</td>
      <td>10</td>
      <td>7</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>37.0</td>
      <td>Yes</td>
      <td>Travel_Rarely</td>
      <td>1373</td>
      <td>Research &amp; Development</td>
      <td>2</td>
      <td>2</td>
      <td>Other</td>
      <td>1</td>
      <td>4</td>
      <td>...</td>
      <td>2</td>
      <td>80</td>
      <td>0</td>
      <td>7</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33.0</td>
      <td>No</td>
      <td>Travel_Frequently</td>
      <td>1392</td>
      <td>Research &amp; Development</td>
      <td>3</td>
      <td>4</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>5</td>
      <td>...</td>
      <td>3</td>
      <td>80</td>
      <td>0</td>
      <td>8</td>
      <td>3</td>
      <td>3</td>
      <td>8</td>
      <td>7</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>27.0</td>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>591</td>
      <td>Research &amp; Development</td>
      <td>2</td>
      <td>1</td>
      <td>Medical</td>
      <td>1</td>
      <td>7</td>
      <td>...</td>
      <td>4</td>
      <td>80</td>
      <td>1</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1469</th>
      <td>37.0</td>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>866</td>
      <td>Sales</td>
      <td>5</td>
      <td>3</td>
      <td>Medical</td>
      <td>1</td>
      <td>1473</td>
      <td>...</td>
      <td>4</td>
      <td>80</td>
      <td>0</td>
      <td>6</td>
      <td>4</td>
      <td>3</td>
      <td>5</td>
      <td>4</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1470</th>
      <td>37.0</td>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>1084</td>
      <td>Research &amp; Development</td>
      <td>13</td>
      <td>2</td>
      <td>Medical</td>
      <td>1</td>
      <td>1474</td>
      <td>...</td>
      <td>3</td>
      <td>80</td>
      <td>2</td>
      <td>5</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1471</th>
      <td>37.0</td>
      <td>Yes</td>
      <td>Travel_Rarely</td>
      <td>240</td>
      <td>Human Resources</td>
      <td>22</td>
      <td>1</td>
      <td>Human Resources</td>
      <td>1</td>
      <td>1475</td>
      <td>...</td>
      <td>3</td>
      <td>80</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1472</th>
      <td>37.0</td>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>1339</td>
      <td>Research &amp; Development</td>
      <td>7</td>
      <td>3</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>1476</td>
      <td>...</td>
      <td>3</td>
      <td>80</td>
      <td>1</td>
      <td>25</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1473</th>
      <td>37.0</td>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>1396</td>
      <td>Research &amp; Development</td>
      <td>5</td>
      <td>2</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>1477</td>
      <td>...</td>
      <td>4</td>
      <td>80</td>
      <td>0</td>
      <td>16</td>
      <td>3</td>
      <td>4</td>
      <td>13</td>
      <td>11</td>
      <td>3</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
<p>1474 rows × 35 columns</p>
</div>




```python
#re-check missing values
df_clean.isnull().sum()
```




    Age                         0
    Attrition                   0
    BusinessTravel              0
    DailyRate                   0
    Department                  0
    DistanceFromHome            0
    Education                   0
    EducationField              0
    EmployeeCount               0
    EmployeeNumber              0
    EnvironmentSatisfaction     0
    Gender                      0
    HourlyRate                  0
    JobInvolvement              0
    JobLevel                    0
    JobRole                     0
    JobSatisfaction             0
    MaritalStatus               0
    MonthlyIncome               0
    MonthlyRate                 0
    NumCompaniesWorked          0
    Over18                      0
    OverTime                    0
    PercentSalaryHike           0
    PerformanceRating           0
    RelationshipSatisfaction    0
    StandardHours               0
    StockOptionLevel            0
    TotalWorkingYears           0
    TrainingTimesLastYear       0
    WorkLifeBalance             0
    YearsAtCompany              0
    YearsInCurrentRole          0
    YearsSinceLastPromotion     0
    YearsWithCurrManager        0
    dtype: int64



## Incorrect Format and Data Type


```python
df_clean.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1474 entries, 0 to 1473
    Data columns (total 35 columns):
     #   Column                    Non-Null Count  Dtype  
    ---  ------                    --------------  -----  
     0   Age                       1474 non-null   float64
     1   Attrition                 1474 non-null   object 
     2   BusinessTravel            1474 non-null   object 
     3   DailyRate                 1474 non-null   int64  
     4   Department                1474 non-null   object 
     5   DistanceFromHome          1474 non-null   int64  
     6   Education                 1474 non-null   int64  
     7   EducationField            1474 non-null   object 
     8   EmployeeCount             1474 non-null   int64  
     9   EmployeeNumber            1474 non-null   int64  
     10  EnvironmentSatisfaction   1474 non-null   int64  
     11  Gender                    1474 non-null   object 
     12  HourlyRate                1474 non-null   int64  
     13  JobInvolvement            1474 non-null   int64  
     14  JobLevel                  1474 non-null   int64  
     15  JobRole                   1474 non-null   object 
     16  JobSatisfaction           1474 non-null   int64  
     17  MaritalStatus             1474 non-null   object 
     18  MonthlyIncome             1474 non-null   int64  
     19  MonthlyRate               1474 non-null   int64  
     20  NumCompaniesWorked        1474 non-null   int64  
     21  Over18                    1474 non-null   object 
     22  OverTime                  1474 non-null   object 
     23  PercentSalaryHike         1474 non-null   int64  
     24  PerformanceRating         1474 non-null   int64  
     25  RelationshipSatisfaction  1474 non-null   int64  
     26  StandardHours             1474 non-null   int64  
     27  StockOptionLevel          1474 non-null   int64  
     28  TotalWorkingYears         1474 non-null   int64  
     29  TrainingTimesLastYear     1474 non-null   int64  
     30  WorkLifeBalance           1474 non-null   int64  
     31  YearsAtCompany            1474 non-null   int64  
     32  YearsInCurrentRole        1474 non-null   int64  
     33  YearsSinceLastPromotion   1474 non-null   int64  
     34  YearsWithCurrManager      1474 non-null   int64  
    dtypes: float64(1), int64(25), object(9)
    memory usage: 403.2+ KB
    

### Change data type 'Age'



```python
df_clean['Age'] = df_clean['Age'].astype('Int64')
```

### Unique values for categorical columns


```python
for col in df_clean.describe(include = 'object').columns:
    print(col)
    print(df_clean[col].unique())
    print('-' * 70)
```

    Attrition
    ['Yes' 'No']
    ----------------------------------------------------------------------
    BusinessTravel
    ['Travel_Rarely' 'Travel_Frequently' 'Travel_Freque' 'Non-Travel']
    ----------------------------------------------------------------------
    Department
    ['Sales' 'Research & Development' 'Reseach & Development'
     'Human Resources']
    ----------------------------------------------------------------------
    EducationField
    ['Life Sciences' 'Other' 'Medical' 'Marketing' 'Technical Degree'
     'Human Resources']
    ----------------------------------------------------------------------
    Gender
    ['Female' 'Male' 'Mle']
    ----------------------------------------------------------------------
    JobRole
    ['Sales Executive' 'Research Scientist' 'Laboratory Technician'
     'Manufacturing Director' 'Healthcare Representative' 'Manager'
     'Sales Representative' 'Research Director' 'Human Resources']
    ----------------------------------------------------------------------
    MaritalStatus
    ['Single' 'Married' 'Divorced']
    ----------------------------------------------------------------------
    Over18
    ['Y']
    ----------------------------------------------------------------------
    OverTime
    ['Yes' 'No']
    ----------------------------------------------------------------------
    

#### Gender Category



```python
#convert 'Mle' to 'Male'
df_clean['Gender'] = df_clean['Gender'].str.replace('Mle', 'Male')
print(df_clean['Gender'].unique())

```

    ['Female' 'Male']
    

#### Business Travel Category


```python
df_clean['BusinessTravel'] = df_clean['BusinessTravel'].str.replace('Travel_Freque', 'Travel_Frequently')

```


```python
df_clean['BusinessTravel'].unique()
```




    array(['Travel_Rarely', 'Travel_Frequentlyntly', 'Travel_Frequently',
           'Non-Travel'], dtype=object)




```python
df_clean['BusinessTravel'] = df_clean['BusinessTravel'].str.replace('Travel_Frequentlyntly', 'Travel_Frequently')
df_clean['BusinessTravel'].unique().tolist()
```




    ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel']



#### Department Category


```python
#replace 'Reseach & Development' with 'Research & Development'
df_clean['Department'] = df_clean['Department'].str.replace('Reseach & Development', 'Research & Development')
df_clean['Department'].unique().tolist()
```




    ['Sales', 'Research & Development', 'Human Resources']



### Filter Total Working Years with positive value only


```python
df_clean = df_clean[df_clean['TotalWorkingYears'] >= 0]
```

## Duplication


```python
#check duplication
print(df_clean.duplicated().sum())
```

    0
    


```python
#delete duplicates
df_clean = df_clean.drop_duplicates()
```


```python
#re-check duplication
print(df_clean.duplicated().sum())
```

    0
    


```python
df_clean.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 1473 entries, 0 to 1473
    Data columns (total 35 columns):
     #   Column                    Non-Null Count  Dtype 
    ---  ------                    --------------  ----- 
     0   Age                       1473 non-null   Int64 
     1   Attrition                 1473 non-null   object
     2   BusinessTravel            1473 non-null   object
     3   DailyRate                 1473 non-null   int64 
     4   Department                1473 non-null   object
     5   DistanceFromHome          1473 non-null   int64 
     6   Education                 1473 non-null   int64 
     7   EducationField            1473 non-null   object
     8   EmployeeCount             1473 non-null   int64 
     9   EmployeeNumber            1473 non-null   int64 
     10  EnvironmentSatisfaction   1473 non-null   int64 
     11  Gender                    1473 non-null   object
     12  HourlyRate                1473 non-null   int64 
     13  JobInvolvement            1473 non-null   int64 
     14  JobLevel                  1473 non-null   int64 
     15  JobRole                   1473 non-null   object
     16  JobSatisfaction           1473 non-null   int64 
     17  MaritalStatus             1473 non-null   object
     18  MonthlyIncome             1473 non-null   int64 
     19  MonthlyRate               1473 non-null   int64 
     20  NumCompaniesWorked        1473 non-null   int64 
     21  Over18                    1473 non-null   object
     22  OverTime                  1473 non-null   object
     23  PercentSalaryHike         1473 non-null   int64 
     24  PerformanceRating         1473 non-null   int64 
     25  RelationshipSatisfaction  1473 non-null   int64 
     26  StandardHours             1473 non-null   int64 
     27  StockOptionLevel          1473 non-null   int64 
     28  TotalWorkingYears         1473 non-null   int64 
     29  TrainingTimesLastYear     1473 non-null   int64 
     30  WorkLifeBalance           1473 non-null   int64 
     31  YearsAtCompany            1473 non-null   int64 
     32  YearsInCurrentRole        1473 non-null   int64 
     33  YearsSinceLastPromotion   1473 non-null   int64 
     34  YearsWithCurrManager      1473 non-null   int64 
    dtypes: Int64(1), int64(25), object(9)
    memory usage: 415.7+ KB
    

## Outlier Detection and Removal


```python
df_clean.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>DailyRate</th>
      <th>DistanceFromHome</th>
      <th>Education</th>
      <th>EmployeeCount</th>
      <th>EmployeeNumber</th>
      <th>EnvironmentSatisfaction</th>
      <th>HourlyRate</th>
      <th>JobInvolvement</th>
      <th>JobLevel</th>
      <th>...</th>
      <th>RelationshipSatisfaction</th>
      <th>StandardHours</th>
      <th>StockOptionLevel</th>
      <th>TotalWorkingYears</th>
      <th>TrainingTimesLastYear</th>
      <th>WorkLifeBalance</th>
      <th>YearsAtCompany</th>
      <th>YearsInCurrentRole</th>
      <th>YearsSinceLastPromotion</th>
      <th>YearsWithCurrManager</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1473.0</td>
      <td>1473.000000</td>
      <td>1473.000000</td>
      <td>1473.000000</td>
      <td>1473.0</td>
      <td>1473.000000</td>
      <td>1473.000000</td>
      <td>1473.000000</td>
      <td>1473.000000</td>
      <td>1473.000000</td>
      <td>...</td>
      <td>1473.000000</td>
      <td>1473.0</td>
      <td>1473.000000</td>
      <td>1473.000000</td>
      <td>1473.000000</td>
      <td>1473.000000</td>
      <td>1473.000000</td>
      <td>1473.000000</td>
      <td>1473.000000</td>
      <td>1473.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>36.928038</td>
      <td>802.702648</td>
      <td>9.183978</td>
      <td>2.909708</td>
      <td>1.0</td>
      <td>740.982349</td>
      <td>2.726409</td>
      <td>65.882553</td>
      <td>2.728445</td>
      <td>2.063815</td>
      <td>...</td>
      <td>2.714189</td>
      <td>80.0</td>
      <td>0.793618</td>
      <td>11.281738</td>
      <td>2.799050</td>
      <td>2.762390</td>
      <td>7.000000</td>
      <td>4.223354</td>
      <td>2.181263</td>
      <td>4.116090</td>
    </tr>
    <tr>
      <th>std</th>
      <td>9.099106</td>
      <td>403.683493</td>
      <td>8.099967</td>
      <td>1.024493</td>
      <td>0.0</td>
      <td>425.393219</td>
      <td>1.093068</td>
      <td>20.308401</td>
      <td>0.711598</td>
      <td>1.105815</td>
      <td>...</td>
      <td>1.081173</td>
      <td>0.0</td>
      <td>0.852291</td>
      <td>7.788303</td>
      <td>1.286563</td>
      <td>0.706548</td>
      <td>6.126275</td>
      <td>3.625423</td>
      <td>3.216445</td>
      <td>3.567207</td>
    </tr>
    <tr>
      <th>min</th>
      <td>18.0</td>
      <td>102.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>30.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>80.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>30.0</td>
      <td>465.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>1.0</td>
      <td>373.000000</td>
      <td>2.000000</td>
      <td>48.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>2.000000</td>
      <td>80.0</td>
      <td>0.000000</td>
      <td>6.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>36.0</td>
      <td>802.000000</td>
      <td>7.000000</td>
      <td>3.000000</td>
      <td>1.0</td>
      <td>741.000000</td>
      <td>3.000000</td>
      <td>66.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>...</td>
      <td>3.000000</td>
      <td>80.0</td>
      <td>1.000000</td>
      <td>10.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>5.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>43.0</td>
      <td>1157.000000</td>
      <td>14.000000</td>
      <td>4.000000</td>
      <td>1.0</td>
      <td>1109.000000</td>
      <td>4.000000</td>
      <td>83.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>...</td>
      <td>4.000000</td>
      <td>80.0</td>
      <td>1.000000</td>
      <td>15.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>9.000000</td>
      <td>7.000000</td>
      <td>3.000000</td>
      <td>7.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>60.0</td>
      <td>1499.000000</td>
      <td>29.000000</td>
      <td>5.000000</td>
      <td>1.0</td>
      <td>1477.000000</td>
      <td>4.000000</td>
      <td>100.000000</td>
      <td>4.000000</td>
      <td>5.000000</td>
      <td>...</td>
      <td>4.000000</td>
      <td>80.0</td>
      <td>3.000000</td>
      <td>40.000000</td>
      <td>6.000000</td>
      <td>4.000000</td>
      <td>40.000000</td>
      <td>18.000000</td>
      <td>15.000000</td>
      <td>17.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 26 columns</p>
</div>




```python
df_clean['MonthlyIncome'].describe()
```




    count     1473.000000
    mean      6501.703327
    std       4704.207326
    min       1009.000000
    25%       2911.000000
    50%       4930.000000
    75%       8380.000000
    max      19999.000000
    Name: MonthlyIncome, dtype: float64




```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


plt.figure(figsize=(10, 6))
plt.boxplot(df_clean['MonthlyIncome'])
plt.title('Box Plot of MonthlyIncome')
plt.ylabel('MonthlyIncome')
plt.show()
```


    
![png](output_32_0.png)
    


based on boxplot, there are some outliers detected.


```python
def outliers(x):
  Q1 = x.quantile(0.25)
  Q3 = x.quantile(0.75)
  IQR = Q3 - Q1

  lower_bound = Q1 - (1.5 * IQR)
  upper_bound = Q3 + (1.5 * IQR)

  return (x < lower_bound) | (x > upper_bound)

outliers_mask = outliers(df_clean['MonthlyIncome'])
outliers_detected = df_clean[outliers_mask]
outliers_detected
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Attrition</th>
      <th>BusinessTravel</th>
      <th>DailyRate</th>
      <th>Department</th>
      <th>DistanceFromHome</th>
      <th>Education</th>
      <th>EducationField</th>
      <th>EmployeeCount</th>
      <th>EmployeeNumber</th>
      <th>...</th>
      <th>RelationshipSatisfaction</th>
      <th>StandardHours</th>
      <th>StockOptionLevel</th>
      <th>TotalWorkingYears</th>
      <th>TrainingTimesLastYear</th>
      <th>WorkLifeBalance</th>
      <th>YearsAtCompany</th>
      <th>YearsInCurrentRole</th>
      <th>YearsSinceLastPromotion</th>
      <th>YearsWithCurrManager</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>24</th>
      <td>53</td>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>1282</td>
      <td>Research &amp; Development</td>
      <td>5</td>
      <td>3</td>
      <td>Other</td>
      <td>1</td>
      <td>28</td>
      <td>...</td>
      <td>4</td>
      <td>80</td>
      <td>1</td>
      <td>26</td>
      <td>3</td>
      <td>2</td>
      <td>14</td>
      <td>13</td>
      <td>4</td>
      <td>8</td>
    </tr>
    <tr>
      <th>28</th>
      <td>46</td>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>705</td>
      <td>Sales</td>
      <td>2</td>
      <td>4</td>
      <td>Marketing</td>
      <td>1</td>
      <td>32</td>
      <td>...</td>
      <td>4</td>
      <td>80</td>
      <td>0</td>
      <td>22</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>44</th>
      <td>41</td>
      <td>Yes</td>
      <td>Travel_Rarely</td>
      <td>1360</td>
      <td>Research &amp; Development</td>
      <td>12</td>
      <td>3</td>
      <td>Technical Degree</td>
      <td>1</td>
      <td>48</td>
      <td>...</td>
      <td>4</td>
      <td>80</td>
      <td>0</td>
      <td>23</td>
      <td>0</td>
      <td>3</td>
      <td>22</td>
      <td>15</td>
      <td>15</td>
      <td>8</td>
    </tr>
    <tr>
      <th>61</th>
      <td>50</td>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>989</td>
      <td>Research &amp; Development</td>
      <td>7</td>
      <td>2</td>
      <td>Medical</td>
      <td>1</td>
      <td>65</td>
      <td>...</td>
      <td>4</td>
      <td>80</td>
      <td>1</td>
      <td>29</td>
      <td>2</td>
      <td>2</td>
      <td>27</td>
      <td>3</td>
      <td>13</td>
      <td>8</td>
    </tr>
    <tr>
      <th>104</th>
      <td>59</td>
      <td>No</td>
      <td>Non-Travel</td>
      <td>1420</td>
      <td>Human Resources</td>
      <td>2</td>
      <td>4</td>
      <td>Human Resources</td>
      <td>1</td>
      <td>108</td>
      <td>...</td>
      <td>4</td>
      <td>80</td>
      <td>1</td>
      <td>30</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1373</th>
      <td>58</td>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>605</td>
      <td>Sales</td>
      <td>21</td>
      <td>3</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>1377</td>
      <td>...</td>
      <td>3</td>
      <td>80</td>
      <td>1</td>
      <td>29</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1376</th>
      <td>49</td>
      <td>No</td>
      <td>Travel_Frequently</td>
      <td>1064</td>
      <td>Research &amp; Development</td>
      <td>2</td>
      <td>1</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>1380</td>
      <td>...</td>
      <td>4</td>
      <td>80</td>
      <td>0</td>
      <td>28</td>
      <td>3</td>
      <td>3</td>
      <td>5</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1400</th>
      <td>55</td>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>189</td>
      <td>Human Resources</td>
      <td>26</td>
      <td>4</td>
      <td>Human Resources</td>
      <td>1</td>
      <td>1404</td>
      <td>...</td>
      <td>1</td>
      <td>80</td>
      <td>1</td>
      <td>35</td>
      <td>0</td>
      <td>3</td>
      <td>10</td>
      <td>9</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1436</th>
      <td>39</td>
      <td>No</td>
      <td>Non-Travel</td>
      <td>105</td>
      <td>Research &amp; Development</td>
      <td>9</td>
      <td>3</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>1440</td>
      <td>...</td>
      <td>3</td>
      <td>80</td>
      <td>0</td>
      <td>21</td>
      <td>3</td>
      <td>2</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1442</th>
      <td>42</td>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>300</td>
      <td>Research &amp; Development</td>
      <td>2</td>
      <td>3</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>1446</td>
      <td>...</td>
      <td>1</td>
      <td>80</td>
      <td>0</td>
      <td>24</td>
      <td>2</td>
      <td>2</td>
      <td>22</td>
      <td>6</td>
      <td>4</td>
      <td>14</td>
    </tr>
  </tbody>
</table>
<p>114 rows × 35 columns</p>
</div>



delete outliers


```python
df_clean = df_clean[~outliers_mask]
df_clean
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Attrition</th>
      <th>BusinessTravel</th>
      <th>DailyRate</th>
      <th>Department</th>
      <th>DistanceFromHome</th>
      <th>Education</th>
      <th>EducationField</th>
      <th>EmployeeCount</th>
      <th>EmployeeNumber</th>
      <th>...</th>
      <th>RelationshipSatisfaction</th>
      <th>StandardHours</th>
      <th>StockOptionLevel</th>
      <th>TotalWorkingYears</th>
      <th>TrainingTimesLastYear</th>
      <th>WorkLifeBalance</th>
      <th>YearsAtCompany</th>
      <th>YearsInCurrentRole</th>
      <th>YearsSinceLastPromotion</th>
      <th>YearsWithCurrManager</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>41</td>
      <td>Yes</td>
      <td>Travel_Rarely</td>
      <td>1102</td>
      <td>Sales</td>
      <td>1</td>
      <td>2</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>80</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>49</td>
      <td>No</td>
      <td>Travel_Frequently</td>
      <td>279</td>
      <td>Research &amp; Development</td>
      <td>8</td>
      <td>1</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>2</td>
      <td>...</td>
      <td>4</td>
      <td>80</td>
      <td>1</td>
      <td>10</td>
      <td>3</td>
      <td>3</td>
      <td>10</td>
      <td>7</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>37</td>
      <td>Yes</td>
      <td>Travel_Rarely</td>
      <td>1373</td>
      <td>Research &amp; Development</td>
      <td>2</td>
      <td>2</td>
      <td>Other</td>
      <td>1</td>
      <td>4</td>
      <td>...</td>
      <td>2</td>
      <td>80</td>
      <td>0</td>
      <td>7</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33</td>
      <td>No</td>
      <td>Travel_Frequently</td>
      <td>1392</td>
      <td>Research &amp; Development</td>
      <td>3</td>
      <td>4</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>5</td>
      <td>...</td>
      <td>3</td>
      <td>80</td>
      <td>0</td>
      <td>8</td>
      <td>3</td>
      <td>3</td>
      <td>8</td>
      <td>7</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>27</td>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>591</td>
      <td>Research &amp; Development</td>
      <td>2</td>
      <td>1</td>
      <td>Medical</td>
      <td>1</td>
      <td>7</td>
      <td>...</td>
      <td>4</td>
      <td>80</td>
      <td>1</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1469</th>
      <td>37</td>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>866</td>
      <td>Sales</td>
      <td>5</td>
      <td>3</td>
      <td>Medical</td>
      <td>1</td>
      <td>1473</td>
      <td>...</td>
      <td>4</td>
      <td>80</td>
      <td>0</td>
      <td>6</td>
      <td>4</td>
      <td>3</td>
      <td>5</td>
      <td>4</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1470</th>
      <td>37</td>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>1084</td>
      <td>Research &amp; Development</td>
      <td>13</td>
      <td>2</td>
      <td>Medical</td>
      <td>1</td>
      <td>1474</td>
      <td>...</td>
      <td>3</td>
      <td>80</td>
      <td>2</td>
      <td>5</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1471</th>
      <td>37</td>
      <td>Yes</td>
      <td>Travel_Rarely</td>
      <td>240</td>
      <td>Human Resources</td>
      <td>22</td>
      <td>1</td>
      <td>Human Resources</td>
      <td>1</td>
      <td>1475</td>
      <td>...</td>
      <td>3</td>
      <td>80</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1472</th>
      <td>37</td>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>1339</td>
      <td>Research &amp; Development</td>
      <td>7</td>
      <td>3</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>1476</td>
      <td>...</td>
      <td>3</td>
      <td>80</td>
      <td>1</td>
      <td>25</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1473</th>
      <td>37</td>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>1396</td>
      <td>Research &amp; Development</td>
      <td>5</td>
      <td>2</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>1477</td>
      <td>...</td>
      <td>4</td>
      <td>80</td>
      <td>0</td>
      <td>16</td>
      <td>3</td>
      <td>4</td>
      <td>13</td>
      <td>11</td>
      <td>3</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
<p>1359 rows × 35 columns</p>
</div>




```python
plt.boxplot(df_clean['MonthlyIncome'])
plt.show()
```


    
![png](output_37_0.png)
    


## Dropping Unnecessary Column


```python
df_clean = df_clean.drop(columns = ['EmployeeCount', 'Over18'])
```


```python
df_clean.columns
```




    Index(['Age', 'Attrition', 'BusinessTravel', 'DailyRate', 'Department',
           'DistanceFromHome', 'Education', 'EducationField', 'EmployeeNumber',
           'EnvironmentSatisfaction', 'Gender', 'HourlyRate', 'JobInvolvement',
           'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus',
           'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'OverTime',
           'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
           'StandardHours', 'StockOptionLevel', 'TotalWorkingYears',
           'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
           'YearsInCurrentRole', 'YearsSinceLastPromotion',
           'YearsWithCurrManager'],
          dtype='object')



## Data Encoding

encoding job satisfaction categories into descriptive labels


```python
satisfaction_map = {
    1: 'Very Dissatisfied',
    2: 'Dissatisfied',
    3: 'Satisfied',
    4: 'Very Satisfied'
}

df_clean['Job_Satisfaction_Cat'] = df_clean['JobSatisfaction'].map(satisfaction_map)
df_clean
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Attrition</th>
      <th>BusinessTravel</th>
      <th>DailyRate</th>
      <th>Department</th>
      <th>DistanceFromHome</th>
      <th>Education</th>
      <th>EducationField</th>
      <th>EmployeeNumber</th>
      <th>EnvironmentSatisfaction</th>
      <th>...</th>
      <th>StandardHours</th>
      <th>StockOptionLevel</th>
      <th>TotalWorkingYears</th>
      <th>TrainingTimesLastYear</th>
      <th>WorkLifeBalance</th>
      <th>YearsAtCompany</th>
      <th>YearsInCurrentRole</th>
      <th>YearsSinceLastPromotion</th>
      <th>YearsWithCurrManager</th>
      <th>Job_Satisfaction_Cat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>41</td>
      <td>Yes</td>
      <td>Travel_Rarely</td>
      <td>1102</td>
      <td>Sales</td>
      <td>1</td>
      <td>2</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>2</td>
      <td>...</td>
      <td>80</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
      <td>Very Satisfied</td>
    </tr>
    <tr>
      <th>1</th>
      <td>49</td>
      <td>No</td>
      <td>Travel_Frequently</td>
      <td>279</td>
      <td>Research &amp; Development</td>
      <td>8</td>
      <td>1</td>
      <td>Life Sciences</td>
      <td>2</td>
      <td>3</td>
      <td>...</td>
      <td>80</td>
      <td>1</td>
      <td>10</td>
      <td>3</td>
      <td>3</td>
      <td>10</td>
      <td>7</td>
      <td>1</td>
      <td>7</td>
      <td>Dissatisfied</td>
    </tr>
    <tr>
      <th>2</th>
      <td>37</td>
      <td>Yes</td>
      <td>Travel_Rarely</td>
      <td>1373</td>
      <td>Research &amp; Development</td>
      <td>2</td>
      <td>2</td>
      <td>Other</td>
      <td>4</td>
      <td>4</td>
      <td>...</td>
      <td>80</td>
      <td>0</td>
      <td>7</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Satisfied</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33</td>
      <td>No</td>
      <td>Travel_Frequently</td>
      <td>1392</td>
      <td>Research &amp; Development</td>
      <td>3</td>
      <td>4</td>
      <td>Life Sciences</td>
      <td>5</td>
      <td>4</td>
      <td>...</td>
      <td>80</td>
      <td>0</td>
      <td>8</td>
      <td>3</td>
      <td>3</td>
      <td>8</td>
      <td>7</td>
      <td>3</td>
      <td>0</td>
      <td>Satisfied</td>
    </tr>
    <tr>
      <th>4</th>
      <td>27</td>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>591</td>
      <td>Research &amp; Development</td>
      <td>2</td>
      <td>1</td>
      <td>Medical</td>
      <td>7</td>
      <td>1</td>
      <td>...</td>
      <td>80</td>
      <td>1</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>Dissatisfied</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1469</th>
      <td>37</td>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>866</td>
      <td>Sales</td>
      <td>5</td>
      <td>3</td>
      <td>Medical</td>
      <td>1473</td>
      <td>4</td>
      <td>...</td>
      <td>80</td>
      <td>0</td>
      <td>6</td>
      <td>4</td>
      <td>3</td>
      <td>5</td>
      <td>4</td>
      <td>1</td>
      <td>3</td>
      <td>Very Dissatisfied</td>
    </tr>
    <tr>
      <th>1470</th>
      <td>37</td>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>1084</td>
      <td>Research &amp; Development</td>
      <td>13</td>
      <td>2</td>
      <td>Medical</td>
      <td>1474</td>
      <td>4</td>
      <td>...</td>
      <td>80</td>
      <td>2</td>
      <td>5</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>Very Dissatisfied</td>
    </tr>
    <tr>
      <th>1471</th>
      <td>37</td>
      <td>Yes</td>
      <td>Travel_Rarely</td>
      <td>240</td>
      <td>Human Resources</td>
      <td>22</td>
      <td>1</td>
      <td>Human Resources</td>
      <td>1475</td>
      <td>4</td>
      <td>...</td>
      <td>80</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Satisfied</td>
    </tr>
    <tr>
      <th>1472</th>
      <td>37</td>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>1339</td>
      <td>Research &amp; Development</td>
      <td>7</td>
      <td>3</td>
      <td>Life Sciences</td>
      <td>1476</td>
      <td>2</td>
      <td>...</td>
      <td>80</td>
      <td>1</td>
      <td>25</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Very Dissatisfied</td>
    </tr>
    <tr>
      <th>1473</th>
      <td>37</td>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>1396</td>
      <td>Research &amp; Development</td>
      <td>5</td>
      <td>2</td>
      <td>Life Sciences</td>
      <td>1477</td>
      <td>4</td>
      <td>...</td>
      <td>80</td>
      <td>0</td>
      <td>16</td>
      <td>3</td>
      <td>4</td>
      <td>13</td>
      <td>11</td>
      <td>3</td>
      <td>7</td>
      <td>Dissatisfied</td>
    </tr>
  </tbody>
</table>
<p>1359 rows × 34 columns</p>
</div>



# Employee Attrition Analysis


```python
df_clean.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 1359 entries, 0 to 1473
    Data columns (total 34 columns):
     #   Column                    Non-Null Count  Dtype 
    ---  ------                    --------------  ----- 
     0   Age                       1359 non-null   Int64 
     1   Attrition                 1359 non-null   object
     2   BusinessTravel            1359 non-null   object
     3   DailyRate                 1359 non-null   int64 
     4   Department                1359 non-null   object
     5   DistanceFromHome          1359 non-null   int64 
     6   Education                 1359 non-null   int64 
     7   EducationField            1359 non-null   object
     8   EmployeeNumber            1359 non-null   int64 
     9   EnvironmentSatisfaction   1359 non-null   int64 
     10  Gender                    1359 non-null   object
     11  HourlyRate                1359 non-null   int64 
     12  JobInvolvement            1359 non-null   int64 
     13  JobLevel                  1359 non-null   int64 
     14  JobRole                   1359 non-null   object
     15  JobSatisfaction           1359 non-null   int64 
     16  MaritalStatus             1359 non-null   object
     17  MonthlyIncome             1359 non-null   int64 
     18  MonthlyRate               1359 non-null   int64 
     19  NumCompaniesWorked        1359 non-null   int64 
     20  OverTime                  1359 non-null   object
     21  PercentSalaryHike         1359 non-null   int64 
     22  PerformanceRating         1359 non-null   int64 
     23  RelationshipSatisfaction  1359 non-null   int64 
     24  StandardHours             1359 non-null   int64 
     25  StockOptionLevel          1359 non-null   int64 
     26  TotalWorkingYears         1359 non-null   int64 
     27  TrainingTimesLastYear     1359 non-null   int64 
     28  WorkLifeBalance           1359 non-null   int64 
     29  YearsAtCompany            1359 non-null   int64 
     30  YearsInCurrentRole        1359 non-null   int64 
     31  YearsSinceLastPromotion   1359 non-null   int64 
     32  YearsWithCurrManager      1359 non-null   int64 
     33  Job_Satisfaction_Cat      1359 non-null   object
    dtypes: Int64(1), int64(24), object(9)
    memory usage: 372.9+ KB
    

## Descriptive Statistics

### Numeric columns


```python
df_clean[['Age', 'DistanceFromHome', 'MonthlyIncome', 'TotalWorkingYears', 'JobSatisfaction']].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>DistanceFromHome</th>
      <th>MonthlyIncome</th>
      <th>TotalWorkingYears</th>
      <th>JobSatisfaction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1359.0</td>
      <td>1359.000000</td>
      <td>1359.000000</td>
      <td>1359.000000</td>
      <td>1359.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>36.044886</td>
      <td>9.341428</td>
      <td>5503.640177</td>
      <td>10.041207</td>
      <td>2.732892</td>
    </tr>
    <tr>
      <th>std</th>
      <td>8.769351</td>
      <td>8.123747</td>
      <td>3317.729214</td>
      <td>6.597166</td>
      <td>1.102962</td>
    </tr>
    <tr>
      <th>min</th>
      <td>18.0</td>
      <td>1.000000</td>
      <td>1009.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>30.0</td>
      <td>2.000000</td>
      <td>2816.000000</td>
      <td>6.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>35.0</td>
      <td>7.000000</td>
      <td>4647.000000</td>
      <td>9.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>41.0</td>
      <td>14.000000</td>
      <td>6811.500000</td>
      <td>13.000000</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>60.0</td>
      <td>29.000000</td>
      <td>16555.000000</td>
      <td>40.000000</td>
      <td>4.000000</td>
    </tr>
  </tbody>
</table>
</div>



### Categorical columns


```python
df_clean.describe(include = 'object')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Attrition</th>
      <th>BusinessTravel</th>
      <th>Department</th>
      <th>EducationField</th>
      <th>Gender</th>
      <th>JobRole</th>
      <th>MaritalStatus</th>
      <th>OverTime</th>
      <th>Job_Satisfaction_Cat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1359</td>
      <td>1359</td>
      <td>1359</td>
      <td>1359</td>
      <td>1359</td>
      <td>1359</td>
      <td>1359</td>
      <td>1359</td>
      <td>1359</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>6</td>
      <td>2</td>
      <td>9</td>
      <td>3</td>
      <td>2</td>
      <td>4</td>
    </tr>
    <tr>
      <th>top</th>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>Research &amp; Development</td>
      <td>Life Sciences</td>
      <td>Male</td>
      <td>Sales Executive</td>
      <td>Married</td>
      <td>No</td>
      <td>Very Satisfied</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>1126</td>
      <td>958</td>
      <td>885</td>
      <td>561</td>
      <td>814</td>
      <td>327</td>
      <td>611</td>
      <td>973</td>
      <td>427</td>
    </tr>
  </tbody>
</table>
</div>



## Attrition Rate by Variable


```python
attrition = df_clean['Attrition'].value_counts().reset_index()
attrition
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Attrition</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>No</td>
      <td>1126</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Yes</td>
      <td>233</td>
    </tr>
  </tbody>
</table>
</div>




```python
import seaborn as sns
sns.countplot(data = df_clean, x = 'Attrition')
plt.title('Attrition Status Count')
plt.show()
```


    
![png](output_53_0.png)
    



```python
att_yes = df_clean['Attrition'].value_counts()['Yes']
att_total = df_clean['Attrition'].count()

attrition_rate = (att_yes/att_total) * 100

print(f'Attrition Rate (Yes) = {attrition_rate:.2f}%')
```

    Attrition Rate (Yes) = 17.14%
    


```python
#encoded Attrition Category
df_clean['Attrition_Numeric'] = df_clean['Attrition'].apply(lambda x: 1 if x =='Yes' else 0)
df_clean['Attrition_Numeric'].head()
```




    0    1
    1    0
    2    1
    3    0
    4    0
    Name: Attrition_Numeric, dtype: int64



### Attrition Rate by JobRole


```python
attrition_role = df_clean.groupby('JobRole')['Attrition_Numeric'].mean().reset_index()
attrition_role.rename(columns = {'Attrition_Numeric' : 'Attrition_Rate'}, inplace = True)

attrition_role.sort_values('Attrition_Rate', ascending = False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>JobRole</th>
      <th>Attrition_Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>Sales Representative</td>
      <td>0.397590</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Human Resources</td>
      <td>0.245283</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Laboratory Technician</td>
      <td>0.239382</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Sales Executive</td>
      <td>0.174312</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Research Scientist</td>
      <td>0.160410</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Manager</td>
      <td>0.071429</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Manufacturing Director</td>
      <td>0.068966</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Healthcare Representative</td>
      <td>0.068702</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Research Director</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
import matplotlib.pyplot as plt

fig = plt.figure(figsize = (6,4))
bars = plt.barh(attrition_role['JobRole'], attrition_role['Attrition_Rate'],
         color =plt.get_cmap('Dark2').colors)
plt.title('Attrition Rate by Job Role')

for bar in bars:
    width = bar.get_width() #nilai attrition rate
    plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, #posisi horizontal, vertikal
             f'{width:.2f}',
             va='center',
             ha='left',
             fontsize=8)
```


    
![png](output_58_0.png)
    


From the bar chart, **Sales Representative have the highest attrition rate**, with a significant **40%** leaving the job, followed by Laboratory Technician and Human Resources.
- I want to know understand the factors influencing this trend, so I'm conducting further EDA:

#### Job Satisfaction and Job Role vs Attrition Rate


```python
role_satis = df_clean.groupby(['JobRole', 'JobSatisfaction'])['Attrition_Numeric'].mean().reset_index()
role_satis.rename(columns = {'Attrition_Numeric' : 'Attrition_Rate'}, inplace = True)
role_satis.sort_values('Attrition_Rate', ascending = False).head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>JobRole</th>
      <th>JobSatisfaction</th>
      <th>Attrition_Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>32</th>
      <td>Sales Representative</td>
      <td>1</td>
      <td>0.583333</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Human Resources</td>
      <td>1</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Sales Representative</td>
      <td>2</td>
      <td>0.476190</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Laboratory Technician</td>
      <td>1</td>
      <td>0.357143</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Sales Representative</td>
      <td>3</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Sales Representative</td>
      <td>4</td>
      <td>0.304348</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Human Resources</td>
      <td>3</td>
      <td>0.285714</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Laboratory Technician</td>
      <td>3</td>
      <td>0.280000</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Research Scientist</td>
      <td>1</td>
      <td>0.236364</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Sales Executive</td>
      <td>1</td>
      <td>0.228571</td>
    </tr>
  </tbody>
</table>
</div>



- From this summary, we observe that the **top 3 job roles with the highest attrition rates** (Sales Rep, Laboratory Tech, Human Resources) also have the lowest job satisfaction **(Very Dissatisfied)**.
- This suggests that **dissatisfaction with their jobs may be a key factor contributing to the high attrition rates** in these roles.

#### Over Time and Job Role vs Attrition Rate


```python
role_time = df_clean.groupby(['JobRole', 'OverTime'])['Attrition_Numeric'].mean().reset_index()
role_time.rename(columns = {'Attrition_Numeric' : 'Attrition_Rate'}, inplace = True)
role_time.sort_values('Attrition_Rate', ascending = False).head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>JobRole</th>
      <th>OverTime</th>
      <th>Attrition_Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>17</th>
      <td>Sales Representative</td>
      <td>Yes</td>
      <td>0.666667</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Laboratory Technician</td>
      <td>Yes</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Human Resources</td>
      <td>Yes</td>
      <td>0.384615</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Research Scientist</td>
      <td>Yes</td>
      <td>0.340206</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Sales Executive</td>
      <td>Yes</td>
      <td>0.329787</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Sales Representative</td>
      <td>No</td>
      <td>0.288136</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Manager</td>
      <td>Yes</td>
      <td>0.222222</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Human Resources</td>
      <td>No</td>
      <td>0.200000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Laboratory Technician</td>
      <td>No</td>
      <td>0.157360</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Sales Executive</td>
      <td>No</td>
      <td>0.111588</td>
    </tr>
  </tbody>
</table>
</div>




```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 6))
sns.barplot(x='Attrition_Rate', y='JobRole', hue='OverTime', #hue untuk membedakan berdasarkan kategori
            data=role_time, palette='viridis')

plt.title('Attrition Rate by Job Role and Overtime Status')
plt.xlabel('Attrition Rate')
plt.ylabel('Job Role')
plt.legend(title='OverTime')
plt.show()

```


    
![png](output_65_0.png)
    


- From this summary, we observe that the **top 3 job roles with the highest attrition rates** (Sales Rep, Laboratory Tech, Human Resources) also have a **higher proportion of employees who work overtime.**
- This suggests that **working overtime may be a key factor contributing to the high attrition rates** in these roles.

### Create Function for Attrition Rate by Category Column


```python
def attrition_rate_by_category(df, column_name):

  attrition_df = df.groupby(column_name)['Attrition_Numeric'].mean().reset_index()
  attrition_df.rename(columns = {'Attrition_Numeric' : 'Attrition_Rate'}, inplace = True)

  attrition_df.sort_values('Attrition_Rate', ascending = False)

#Pie chart
  fig = plt.figure(figsize = (6,4))
  plt.pie(attrition_df['Attrition_Rate'], labels = attrition_df[column_name],
        autopct = '%1.2f%%', colors=plt.get_cmap('Pastel1').colors,
        startangle = 90)
  plt.title(f'Attrition Rate by {column_name}')
```

#### Attrition Rate by Marital Status


```python
attrition_rate_by_category(df_clean, 'MaritalStatus')
```


    
![png](output_70_0.png)
    


#### Attrition Rate by Department


```python
attrition_rate_by_category(df_clean, 'Department')
```


    
![png](output_72_0.png)
    


#### Attrition Rate by Gender


```python
attrition_rate_by_category(df_clean, 'Gender')
```


    
![png](output_74_0.png)
    


## Monthly Income vs Attrition


```python
att_yes = df_clean[df_clean['Attrition'] == 'Yes']
att_no = df_clean[df_clean['Attrition'] == 'No']
```


```python
import seaborn as sns

fig = plt.figure(figsize = (6,4))
sns.boxplot(data = df_clean, x = df_clean['Attrition'], y = df_clean['MonthlyIncome'],color = 'white')
plt.title('Monthly Income by Attrition Status')
plt.show()
```


    
![png](output_77_0.png)
    


From the boxplot, we can see that employees with attrition tend to have a lower monthly income compared to employees without attrition.

## Job Satisfaction vs Attrition


```python
satis_att = df_clean.groupby('Job_Satisfaction_Cat', as_index = False)['Attrition_Numeric'].mean()
satis_att.sort_values('Attrition_Numeric', ascending = False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Job_Satisfaction_Cat</th>
      <th>Attrition_Numeric</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>Very Dissatisfied</td>
      <td>0.244361</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Satisfied</td>
      <td>0.178922</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Dissatisfied</td>
      <td>0.170543</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Very Satisfied</td>
      <td>0.119438</td>
    </tr>
  </tbody>
</table>
</div>




```python
import matplotlib.pyplot as plt

fig = plt.figure(figsize = (6,4))
plt.barh(satis_att['Job_Satisfaction_Cat'], satis_att['Attrition_Numeric'],
         color =plt.get_cmap('Dark2').colors)
plt.title('Attrition Rate by Job Satisfaction')
plt.show()
```


    
![png](output_81_0.png)
    


- Employees who are very dissatisfied with their job have the highest attrition rate (25.4%).
- This suggests that high dissatisfaction is **strongly associated** with a higher likelihood of leaving the company

## Commute Distance vs Attrition


```python
import seaborn as sns
sns.boxplot(df_clean, x = df_clean['Attrition'], y = df_clean['DistanceFromHome'], color = 'white')
plt.title('Commute Distance by Attrition Status')
plt.show()
```


    
![png](output_84_0.png)
    


Employees who have a longer distance from home are more likely to attrite



## Tenure Analysis


```python
#create total working years bin
df_clean = df_clean.copy()
bins = [0, 10, 20, 30, 40]
labels = ['0-10', '11-20', '21-30', '31-40']
df_clean['TotalWorkingYearsBins'] = pd.cut(df_clean['TotalWorkingYears'], bins = bins, labels = labels, include_lowest = True)
df_clean
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Attrition</th>
      <th>BusinessTravel</th>
      <th>DailyRate</th>
      <th>Department</th>
      <th>DistanceFromHome</th>
      <th>Education</th>
      <th>EducationField</th>
      <th>EmployeeNumber</th>
      <th>EnvironmentSatisfaction</th>
      <th>...</th>
      <th>TotalWorkingYears</th>
      <th>TrainingTimesLastYear</th>
      <th>WorkLifeBalance</th>
      <th>YearsAtCompany</th>
      <th>YearsInCurrentRole</th>
      <th>YearsSinceLastPromotion</th>
      <th>YearsWithCurrManager</th>
      <th>Job_Satisfaction_Cat</th>
      <th>Attrition_Numeric</th>
      <th>TotalWorkingYearsBins</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>41</td>
      <td>Yes</td>
      <td>Travel_Rarely</td>
      <td>1102</td>
      <td>Sales</td>
      <td>1</td>
      <td>2</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>2</td>
      <td>...</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
      <td>Very Satisfied</td>
      <td>1</td>
      <td>0-10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>49</td>
      <td>No</td>
      <td>Travel_Frequently</td>
      <td>279</td>
      <td>Research &amp; Development</td>
      <td>8</td>
      <td>1</td>
      <td>Life Sciences</td>
      <td>2</td>
      <td>3</td>
      <td>...</td>
      <td>10</td>
      <td>3</td>
      <td>3</td>
      <td>10</td>
      <td>7</td>
      <td>1</td>
      <td>7</td>
      <td>Dissatisfied</td>
      <td>0</td>
      <td>0-10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>37</td>
      <td>Yes</td>
      <td>Travel_Rarely</td>
      <td>1373</td>
      <td>Research &amp; Development</td>
      <td>2</td>
      <td>2</td>
      <td>Other</td>
      <td>4</td>
      <td>4</td>
      <td>...</td>
      <td>7</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Satisfied</td>
      <td>1</td>
      <td>0-10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33</td>
      <td>No</td>
      <td>Travel_Frequently</td>
      <td>1392</td>
      <td>Research &amp; Development</td>
      <td>3</td>
      <td>4</td>
      <td>Life Sciences</td>
      <td>5</td>
      <td>4</td>
      <td>...</td>
      <td>8</td>
      <td>3</td>
      <td>3</td>
      <td>8</td>
      <td>7</td>
      <td>3</td>
      <td>0</td>
      <td>Satisfied</td>
      <td>0</td>
      <td>0-10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>27</td>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>591</td>
      <td>Research &amp; Development</td>
      <td>2</td>
      <td>1</td>
      <td>Medical</td>
      <td>7</td>
      <td>1</td>
      <td>...</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>Dissatisfied</td>
      <td>0</td>
      <td>0-10</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1469</th>
      <td>37</td>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>866</td>
      <td>Sales</td>
      <td>5</td>
      <td>3</td>
      <td>Medical</td>
      <td>1473</td>
      <td>4</td>
      <td>...</td>
      <td>6</td>
      <td>4</td>
      <td>3</td>
      <td>5</td>
      <td>4</td>
      <td>1</td>
      <td>3</td>
      <td>Very Dissatisfied</td>
      <td>0</td>
      <td>0-10</td>
    </tr>
    <tr>
      <th>1470</th>
      <td>37</td>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>1084</td>
      <td>Research &amp; Development</td>
      <td>13</td>
      <td>2</td>
      <td>Medical</td>
      <td>1474</td>
      <td>4</td>
      <td>...</td>
      <td>5</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>Very Dissatisfied</td>
      <td>0</td>
      <td>0-10</td>
    </tr>
    <tr>
      <th>1471</th>
      <td>37</td>
      <td>Yes</td>
      <td>Travel_Rarely</td>
      <td>240</td>
      <td>Human Resources</td>
      <td>22</td>
      <td>1</td>
      <td>Human Resources</td>
      <td>1475</td>
      <td>4</td>
      <td>...</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Satisfied</td>
      <td>1</td>
      <td>0-10</td>
    </tr>
    <tr>
      <th>1472</th>
      <td>37</td>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>1339</td>
      <td>Research &amp; Development</td>
      <td>7</td>
      <td>3</td>
      <td>Life Sciences</td>
      <td>1476</td>
      <td>2</td>
      <td>...</td>
      <td>25</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Very Dissatisfied</td>
      <td>0</td>
      <td>21-30</td>
    </tr>
    <tr>
      <th>1473</th>
      <td>37</td>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>1396</td>
      <td>Research &amp; Development</td>
      <td>5</td>
      <td>2</td>
      <td>Life Sciences</td>
      <td>1477</td>
      <td>4</td>
      <td>...</td>
      <td>16</td>
      <td>3</td>
      <td>4</td>
      <td>13</td>
      <td>11</td>
      <td>3</td>
      <td>7</td>
      <td>Dissatisfied</td>
      <td>0</td>
      <td>11-20</td>
    </tr>
  </tbody>
</table>
<p>1359 rows × 36 columns</p>
</div>




```python
pd.cut(df_clean['TotalWorkingYears'], bins=bins, labels = labels, include_lowest=True).value_counts()

```




    TotalWorkingYears
    0-10     923
    11-20    339
    21-30     76
    31-40     21
    Name: count, dtype: int64




```python
tenure = df_clean.groupby('TotalWorkingYearsBins')['Attrition_Numeric'].mean()
tenure
```

    C:\Users\ASUS\AppData\Local\Temp\ipykernel_8500\2096106399.py:1: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      tenure = df_clean.groupby('TotalWorkingYearsBins')['Attrition_Numeric'].mean()
    




    TotalWorkingYearsBins
    0-10     0.198267
    11-20    0.115044
    21-30    0.105263
    31-40    0.142857
    Name: Attrition_Numeric, dtype: float64




```python
tenure2 = df_clean.groupby('TotalWorkingYears', as_index = False)['Attrition_Numeric'].mean().round(2)
tenure2.rename(columns = {
    'Attrition_Numeric' : 'Attrition_Rate'},
    inplace = True)

tenure2 = pd.DataFrame(tenure2)
tenure2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TotalWorkingYears</th>
      <th>Attrition_Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.45</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0.29</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0.21</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0.19</td>
    </tr>
  </tbody>
</table>
</div>




```python
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12, 4))
ax = fig.add_subplot()

ax.plot(tenure2['TotalWorkingYears'], tenure2['Attrition_Rate'],
        color='blue', linestyle='-')
ax.set_xlabel('Total Working Years')
ax.set_ylabel('Attrition Rate')
ax.set_title('Attrition Rate vs. Total Working Years')

plt.show()
```


    
![png](output_91_0.png)
    


- There is **decreasing trend in attrition rates for 0-10 years range**, but it still higher compared to the 11-20 and 21-30 years range, which have relatively stable and lower attrition rates
- The attrition rate **peaks significantly for employees in the 30-40 work years** range, indicating a higher likelihood of attrition among older group.

# Summary
1. **Sales Representatives** have the **highest attrition rate** (**39.76%**), followes by Laboratory Technicians (24.03%) and Human Resources (23.08%)
  - These 3 job roles also have the **lowest job satisfaction.**
  - Also have a **higher proportion of employees who work overtime.**
  - This suggest that **dissatisfaction and working overtime could be a significant factor influencing high attrition.**

2. The boxplot analysis indicates that **employees with lower monthly incomes**  and **have longer commute distance** are more likely to attrite.

3. There is a **different trend in working years** for each age range.
  - Negative trend of attrition rates for employees with 0-10 years range but it still higher than 11-30 range
  - Stationer at 11-30 years range
  - Significant peak for employees with 31-50 years experience.



# Recommendations
1. Improve employee retention, such as identify and resolve job disssatisfaction (example by survey employees) or reviewing overtime policies.
2. Consider transportation support to reduce the impact of long commutes on employee retention.
3. Make different approach for each working years range:
  - 0-10 years: focus on improving or providing growth opportunities, such as career development.
  - 11-30 years: maintain engagement by rewarding long-term contributions
  - 31-40 years: implementing targeted retirement planning or flexible work options.
