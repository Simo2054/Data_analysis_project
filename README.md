# Data_analysis_project

This is a project that I've done with a few other colleagues from faculty, where we had to import a set of data and do some data analysis tasks.
<br />
The dataset is imported from https://www.kaggle.com/datasets/nevildhinoja/e-commerce-sales-prediction-dataset 
<br />
<br />
### Analyzes performed:
* reading the csv file using ```pandas``` library
* printing the number of entries
* checking if there are missing values by printing the number of missing values
* creating a new dataframe with the Date column transformed in a datetime format
* breaking the Date column in the original dataframe into three other columns 
(Day, Month, Year)
* distributions per category (Price, Discount, Marketing Spend, Customer Segment)
* distributions on category VS category (Price VS Units Sold, Discount VS Units Sold, Marketing Spend VS Units Sold)
* pair plot for numerical values
* sales trend over time
* aggregated statistics for each numerical column
* aggregated statistics for each numerical column grouped by product categories (and by customer segment)
* quartiles for the Price column
* Shapiro analysis for each category of products
* Levene analysis for each value in Units_Sold column by each category of products
* Shapiro analysis for each segment of clients
* Levene analysis for each value in Units_Sold column by each segment of clients
* deducting categorical values
* correlation on columns from the dataframe
* linear regression and OLS for the dataframe
* transforming the categorical variables in dummy variables
* correlation on columns from the dataframe with dummies
* linear regression and OLS for the dataframe with dummies
* testing if there are duplicates in the same day for the same category
* checking and adding the marketing spends for each category of products
* two way Anova analysis for checking if the units sold by month and product categories have effect on units sold


