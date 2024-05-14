# Project Based Internship ID/X Partners x Rakamin Academy
# Final Task: Minimizing Credit Risk Through Client Classification

This project is a collaboration between IDX Partners and Rakamin Academy in the framework of Project-Based Internship (PBI) for Data Science.

## Project Background
The background of this project is to develop a machine learning model that can predict credit risk. This is motivated by the high number of clients defaulting on their loans.

#### Objective:
Therefore, the main objective of this model is to improve accuracy in assessing and managing credit risk, in order to optimize their business decisions and reduce potential losses.

--------------------------------------------------------------
In working on this project, I went through several stages:
## Business Understanding
The initial step in the CRISP-DM process revolves around gaining a comprehensive understanding of the business problem and objectives at hand. Currently, the company is grappling with a default rate of 11.9%, resulting in substantial financial losses amounting to approximately 790K USD. Consequently, the company is determined to mitigate the default rate, minimize losses, and ultimately enhance profitability. To achieve this, the company is in need of a predictive model capable of evaluating the creditworthiness of borrowers. Their primary goal is to identify borrowers who are more prone to default on their loans, enabling them to make informed decisions regarding loan approvals or denials.

## Data Understanding
I obtained the data from Rakamin and ID/X Partners, and I have conveniently placed it in the Data folder. The dataset comprises of 466,285 rows and 75 columns. In general, the data can be categorized into two main segments: borrower data and loan data.

## Exploratory Data Analysis (EDA)
During this process, I conduct a range of analyses pertaining to both borrower and loan data. My aim is to identify any distinctive patterns that may arise throughout the analysis phase.

To facilitate data analysis, I employ several types of plots and graphs. These include bar plots, line plots, stacked bar plots, pie plots, and box plots. By utilizing these visualizations, I can gain deeper insights and uncover valuable trends within the data.

## Data Preprocessing
Several processes in data preprocessing:
1. Feature creation

2. Drop columns
- With more than 40% missing values
- High Cardinality
- Low Variance
- Multicollinearity
- Unnecessary columns

3. Impute columns
- With less than 40% missing values
- Numerical: median
- Categorical: mode

4. Feature Selection
- Chi square test
- ANOVA test
- Multicollinearity
- Cap Outliers

5. Feature Engineering
- Encoding
- Scaling
- Resampling

## Modeling
There are several models that I use:
- LogisticRegression
- LGBMClassifier
- RandomForestClassifier
- XGBClassifier
- GaussianNB

I will choose the best model by comparing them using the Accuracy, Recall, Precision, F1, AUCROC, and Confusion Matrix.

## Evaluation
### Simulation result:
Total bad loans before: $55,332    
Total bad loans after: $10,980    
Decreased bad loans: -$44,352

Default rate before: 11.9%  
Default rate after: 2.4%  
Decreased default rate: -9.5%

Total Revenue before: $5,873,258,515  
Total Revenue after: $6,507,128,475

Total default loss amount before: $790,793,935  
Total default loss amount after: $156,923,975

Net revenue before: $5,082,464,579  
Net revenue after: $6,350,204,499

Net revenue increase: $1,267,739,920

### Summary:  
- The most optimal model to use in this case is the `XGBClassifier`
- This model has a good performance in classifying “Good Loan” with a high True Positive Rate (TPR) of 87.48%
- 3 features that have a significant influence are term_60months, last_pymnt_amnt, days_since_last_pymnt_d
- By using this model, we can minimize losses due to default and increase profits by $1,267,739,920

## Conclusion
The ideal model for this particular case is the XGBClassifier. Through metric measurements and simulations, it has been demonstrated that this model exhibits outstanding performance and effectively mitigates credit risks, including default. By implementing this model, the company stands to increase its potential for profitability.

---------------------------------------------
## About Company ID/X Partners
id/x partners was established in 2002 by ex-bankers and management consultants who have vast experiences in credit cycle and process management, scoring development, and performance management. Our combined experience has served corporations across Asia and Australia regions and in multiple industries, specifically financial services, telecommunications, manufacturing and retail.

id/x partners provides consulting services that specializes in utilizing data analytic and decisioning (DAD) solutions combined with an integrated risk management and marketing discipline to help clients optimize the portfolio profitability and business process.

Comprehensive consulting service and technology solutions offered by id/x partners makes it as a one-stop service provider.
