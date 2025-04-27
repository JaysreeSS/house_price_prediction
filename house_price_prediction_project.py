!pip install boruta

# Libraries
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, KFold
import seaborn as sns
sns.set()

# Decoding Datatype
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# Splitting Dataset
from sklearn.model_selection import train_test_split

# Evaluation Metrics
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Machine Learning Algorithms
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor

import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
dir='/content/drive/MyDrive/Chennai houseing sale.csv'
df=pd.read_csv(dir)

df.head()

df.columns

df.shape

#Checking null values
df.isnull().sum()

#Removing the null values
df=df.dropna()
df.isnull().sum()

df.nunique()

df.info()

# Checking the labels in categorical features
for col in df.columns:
    if df[col].dtype=='object':
        print()
        print(col)
        print(df[col].unique())

## Converting the datatype of Features from float to integer
df["N_BATHROOM"]=df["N_BATHROOM"].astype(int)
df["N_BATHROOM"]=df["N_BATHROOM"].astype(int)
df["N_BEDROOM"]=df["N_BEDROOM"].astype(int)
df["QS_BATHROOM"]=df["QS_BATHROOM"].astype(int)
df["QS_BEDROOM"]=df["QS_BEDROOM"].astype(int)
df["QS_OVERALL"]=df["QS_OVERALL"].astype(int)
df["QS_ROOMS"]=df["QS_ROOMS"].astype(int)

## Renaming the inappropriately given names
## We use .replace() with .apply() to refine the names
df = df.apply(lambda x: x.replace({'Adyr':'Adyar', 'TNagar': 'T Nagar', 'Chrompt': 'Chrompet', 'Chrmpet':'Chrompet', 'Chormpet': 'Chrompet', 'Ann Nagar': 'Anna Nagar',
                                   'Ana Nagar': 'Anna Nagar', 'Velchery': 'Velachery', 'KKNagar': 'KK Nagar', 'Karapakam':'Karapakkam', 'Ab Normal': 'AbNormal',
                                   'Partiall':'Partial', 'PartiaLl': 'Partial', 'AdjLand': 'Adj Land', 'Noo': 'No',
                                   'Comercial':'Commercial', 'Others': 'Other', 'AllPub': 'All Pub', 'NoSewr ':'NoSeWa', 'NoAccess': 'No Access', 'Pavd':'Paved'}, regex=True))

## As per the above .info() function output, the two date columns - DATE_SALE & DATE_BUILD are in object format
## We convert them into DateTime dtype
df['DATE_SALE'] = pd.to_datetime(df['DATE_SALE'], format='%d-%m-%Y')
df['DATE_BUILD'] = pd.to_datetime(df['DATE_BUILD'], format='%d-%m-%Y')
df[['DATE_SALE', 'DATE_BUILD']].head(5)

## We create a new column to determine the age of property
df['PROP_AGE'] = pd.DatetimeIndex(df['DATE_SALE']).year - pd.DatetimeIndex(df['DATE_BUILD']).year
df['PROP_AGE'].head(5)

## Converting object to int
cat_df = df[['AREA', 'SALE_COND', 'PARK_FACIL', 'BUILDTYPE', 'UTILITY_AVAIL', 'STREET', 'MZZONE']]
num_df = df[['INT_SQFT', 'DIST_MAINROAD', 'N_BEDROOM', 'N_BATHROOM', 'N_ROOM', 'QS_ROOMS','QS_BATHROOM',
             'QS_BEDROOM','QS_OVERALL','REG_FEE','COMMIS','SALES_PRICE','SALES_PRICE', 'PROP_AGE']]

le=LabelEncoder()
#select ctegorical columns
cat_df = df.select_dtypes(exclude=["int", "float"])

for i in cat_df:
        cat_df[i] = le.fit_transform(df[i])

#joining the encoded data to the numeric data
num_df = df.select_dtypes(include=['int', 'float'])
num_df = num_df.drop('SALES_PRICE', axis=1)
main_df = pd.concat([num_df, cat_df, df['SALES_PRICE']], axis=1)

main_df.info()
main_df.shape

main_df.head()

data = main_df.drop('SALES_PRICE',axis=1)

#Splitting x and y as training and testing: 70% and 30%
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size = 0.03, random_state =0)

#random forest method
model_rf = RandomForestRegressor()
model_rf.fit(x_train,y_train)
y_pred=model_rf.predict(x_test)
s = r2_score(y_test, y_pred)
s

"""**1. Feature Selection = Feature Correlation**"""

from yellowbrick.target import FeatureCorrelation
# Create a list of the feature names
visualizer = FeatureCorrelation(labels=data.columns)
visualizer.fit(data, target)        # Fit the data to the visualizer
visualizer.poof()
correlation_matrix = visualizer.features_
target_correlation = pd.Series(correlation_matrix)
feature_importances = model_rf.feature_importances_
feature_info = pd.DataFrame({'Feature': data.columns,
                             'Correlation': target_correlation,
                             'Importance': feature_importances})
ranked_features = feature_info.sort_values(by='Importance', ascending=False)
print(ranked_features)

for col in df.columns:
    if df[col].name in ['REG_FEE', 'AREA', 'INT_SQFT', 'BUILDTYPE', 'MZZONE', 'N_BEDROOM', 'N_BATHROOM', 'COMMIS', 'PARK_FACIL', 'DIST_MAINROAD']:
        print()
        print(col)
        print(df[col].unique())

for col in main_df.columns:
    if main_df[col].name in ['REG_FEE', 'AREA', 'INT_SQFT', 'BUILDTYPE', 'MZZONE', 'N_BEDROOM', 'N_BATHROOM', 'COMMIS', 'PARK_FACIL', 'DIST_MAINROAD']:
        print()
        print(col)
        print(main_df[col].unique())

#Selecting independent variables as inputs according to efficiency
data_selected_fc=main_df[['REG_FEE', 'AREA', 'INT_SQFT', 'BUILDTYPE', 'MZZONE',
                          'N_BEDROOM', 'N_BATHROOM', 'COMMIS', 'PARK_FACIL', 'DIST_MAINROAD']]
data_selected_fc

#Selecting dependent and independent variables
target = main_df['SALES_PRICE']

#Splitting x and y as training and testing: 70% and 30%
x_train1, x_test1, y_train1, y_test1 = train_test_split(data_selected_fc, target, test_size = 0.03, random_state =0)

#random forest regression
model_rf1 = RandomForestRegressor()
model_rf1.fit(x_train1,y_train1)
y_pred1=model_rf1.predict(x_test1)
s1 = r2_score(y_test1, y_pred1)
s1

#linear regression
model_lr = LinearRegression()
model_lr.fit(x_train1,y_train1)
y_pred1=model_lr.predict(x_test1)
s1L = r2_score(y_test1, y_pred1)
s1L

#decision tree regression
model_dtr = DecisionTreeRegressor()
model_dtr.fit(x_train1,y_train1)
y_pred1=model_dtr.predict(x_test1)
s1D = r2_score(y_test1, y_pred1)
s1D

#gradient boosting regression
model_gbr = GradientBoostingRegressor()
model_gbr.fit(x_train1,y_train1)
y_pred1=model_gbr.predict(x_test1)
s1G = r2_score(y_test1, y_pred1)
s1G

import pickle
pickle.dump(model_rf1, open('model_rf.pkl', 'wb'))

"""**2. Feature Selection = Mutual Informataion Regression**"""

from sklearn.feature_selection import mutual_info_regression
mutual_info = mutual_info_regression(data, target)
mutual_info = pd.Series(mutual_info)
mutual_info.index = x_train.columns
mutual_info.sort_values(ascending=False).plot.bar(figsize=(20, 8))

# Create a DataFrame to store feature names and scores
mi_df = pd.DataFrame({'Feature': data.columns, 'Mutual_Information': mutual_info})

# Sort the DataFrame by the mutual information scores in descending order
ranked_features = mi_df.sort_values(by='Mutual_Information', ascending=False)

# Access the highest-ranked feature
highest_ranked_feature = ranked_features.iloc[0]['Feature']

# Print or display the highest-ranked feature and its mutual information score
print("Highest Ranked Feature:", highest_ranked_feature)
print("Mutual Information Score:", ranked_features.iloc[0]['Mutual_Information'])

data_selected_mic=main_df[['REG_FEE', 'N_ROOM', 'INT_SQFT', 'COMMIS', 'BUILDTYPE',
                           'N_BEDROOM', 'AREA', 'MZZONE', 'N_BATHROOM', 'DATE_SALE']]
data_selected_mic

#Splitting x and y as training and testing: 70% and 30%
x_train2, x_test2, y_train2, y_test2 = train_test_split(data_selected_mic, target, test_size = 0.03, random_state =0)

#random forest method
model_rf2 = RandomForestRegressor()
model_rf2.fit(x_train2,y_train2)
y_pred2=model_rf2.predict(x_test2)
s2 = r2_score(y_test2, y_pred2)
s2

"""**3. Feature Selection = LASSO Regression**"""

from sklearn.linear_model import Lasso
names=main_df.drop("SALES_PRICE", axis=1).columns
# calling the model with the best parameter
lasso_model = Lasso(alpha=0.01)
lasso_model.fit(data, target)

coefficients = lasso_model.coef_
feature_names = data.columns

# Create a DataFrame to store coefficients and feature names
coefficients_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})

# Sort the DataFrame by the absolute values of coefficients in descending order
ranked_features = coefficients_df.sort_values(by='Coefficient', key=lambda x: abs(x), ascending=False)

# Access the highest-ranked feature
highest_ranked_feature = ranked_features.iloc[0]['Feature']

# Print or display the highest-ranked feature and its coefficient
print("Highest Ranked Feature:", highest_ranked_feature)
print("Coefficient:", ranked_features.iloc[0]['Coefficient'])

# Optionally, you can plot the coefficients
plt.figure(figsize=(10, 6))
plt.bar(ranked_features['Feature'], ranked_features['Coefficient'])
plt.xticks(rotation=45, ha='right')
plt.title('Lasso Coefficients for Features')
plt.show()

data_selected_lasso=main_df[['N_ROOM', 'N_BEDROOM', 'N_BATHROOM', 'BUILDTYPE', 'PARK_FACIL',
                             'MZZONE', 'AREA', 'STREET', 'UTILITY_AVAIL', 'QS_OVERALL']]
data_selected_lasso

#Splitting x and y as training and testing: 70% and 30%
x_train3, x_test3, y_train3, y_test3 = train_test_split(data_selected_lasso, target, test_size = 0.03, random_state =0)

#random forest method
model_rf3 = RandomForestRegressor()
model_rf3.fit(x_train3,y_train3)
y_pred3=model_rf3.predict(x_test3)
s3 = r2_score(y_test3, y_pred3)
s3

"""**4. Feature Selection = Boruta**"""

from boruta import BorutaPy
X = main_df.iloc[:,:-1].values
y = main_df.iloc[:,-1].values
np.int = np.int32
np.float = np.float64
np.bool = np.bool_
feat_selector = BorutaPy(model_rf, n_estimators='auto', verbose=2, max_iter=25, alpha=0.01)
feat_selector.fit(X, y)
most_important = main_df.columns[:-1][feat_selector.support_].tolist()
print(most_important)

data_selected_boruta = main_df[['INT_SQFT', 'N_BEDROOM', 'N_BATHROOM', 'REG_FEE', 'PRT_ID',
                                'COMMIS', 'AREA', 'PARK_FACIL', 'BUILDTYPE', 'MZZONE']]
data_selected_boruta

#Splitting x and y as training and testing: 70% and 30%
x_train4, x_test4, y_train4, y_test4 = train_test_split(data_selected_boruta, target, test_size = 0.03, random_state =0)

#random forest method
model_rf4 = RandomForestRegressor()
model_rf4.fit(x_train4,y_train4)
y_pred4 = model_rf4.predict(x_test4)
s4 = r2_score(y_test4, y_pred4)
s4

"""**5. Feature Selection = Sequential Feature Selection**"""

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
sfs = SFS(LinearRegression(), k_features=10, forward=True, floating=False,
          scoring='r2', cv=0)
sfs.fit(data, target)
print(sfs.k_feature_names_)

data_selected_sfs = main_df[['N_BEDROOM', 'N_BATHROOM', 'N_ROOM', 'REG_FEE', 'COMMIS',
                             'AREA', 'PARK_FACIL', 'BUILDTYPE', 'STREET', 'MZZONE']]
data_selected_sfs

#Splitting x and y as training and testing: 70% and 30%
x_train5, x_test5, y_train5, y_test5 = train_test_split(data_selected_sfs, target, test_size = 0.03, random_state =0)

#random forest method
model_rf5 = RandomForestRegressor()
model_rf5.fit(x_train5,y_train5)
y_pred5=model_rf5.predict(x_test5)
s5 = r2_score(y_test5, y_pred5)
s5

"""**Finding the common features**"""

common_elements = list(set(list(data)).intersection(set(list(data_selected_fc)),
                                                    set(list(data_selected_mic)),
                                                    set(list(data_selected_lasso)),
                                                    set(list(data_selected_boruta)),
                                                    set(list(data_selected_sfs))))
print(common_elements)

common_elements1 = list(set(list(data_selected_fc)).intersection(set(list(data_selected_boruta))))
print(common_elements1)

"""**Plotting r2_scores**"""

fig, ax = plt.subplots()
x = ['Feature Correlation', 'Boruta', 'Sequential Feature Selector',
     'Mutual Information Regression', 'LASSO Regression']
y = [round(s1,ndigits=5),round(s4,ndigits=5),round(s5,ndigits=5),round(s2,ndigits=5),round(s3,ndigits=5)]
plt.plot(x, y, marker = 'o', markerfacecolor = 'red', markersize = 12)
ax.set_ylabel('R-squared Score')
ax.set_title('R-squared Scores for Different Feature Selection Methods (Random Forest)')
for i in range(len(x)):
        plt.text(i,y[i],y[i], ha='left', va='bottom')
plt.xticks(rotation=90)
plt.show()