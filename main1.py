import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score


df=pd.read_csv("D:\datascience\Dataset .csv")
df.info()

#To read number of columns and rows

print(df.shape[0])#rows
print(df.shape[1])#columns

#Find for any missing values

missing_values=df.isnull().sum()
print(missing_values)

#Handle missing values

df=df.dropna() #remove rows
df=df.dropna(axis=1) #remove columns
print("Data after handling missing values")
print(df.isnull().sum())

#check datset to find any conversion required
df.info()

#Find any class imbalances
class_counts=df['Aggregate rating'].value_counts()
print("class counts:")
print(class_counts)
class_counts.plot(kind='bar')
plt.title('Class Distribution')
plt.xlabel('Classes')
plt.ylabel('Count')
plt.show()
imbalance_ratio=class_counts.max()
print("Imbalance Ratio:",imbalance_ratio)
"""
#Identify numerical coolumns
data_types=df.dtypes
num_columns=data_types[data_types!='object'].index.tolist()
print("Numerical columns are:")
print(num_columns)

#Calculate basic statistics
basic_stats=df[['Restaurant ID', 'Country Code', 'Longitude', 'Latitude', 'Average Cost for two', 'Price range', 'Aggregate rating', 'Votes']].describe()
print(basic_stats)

mean=df[['Restaurant ID', 'Country Code', 'Longitude', 'Latitude', 'Average Cost for two', 'Price range', 'Aggregate rating', 'Votes']].mean()
median=df[['Restaurant ID', 'Country Code', 'Longitude', 'Latitude', 'Average Cost for two', 'Price range', 'Aggregate rating', 'Votes']].median()
SD=df[['Restaurant ID', 'Country Code', 'Longitude', 'Latitude', 'Average Cost for two', 'Price range', 'Aggregate rating', 'Votes']].std()


print("Mean:")
print(mean)

print("Median:")
print(median)
print("standard deviation:")
print(SD)

#Distribution of categorical variables
country_dis=df['Country Code'].value_counts()
print("country code distribution is:")
print(country_dis)

city_dis=df['City'].value_counts()
print("city distribution:")
print(city_dis)

cuisine_dis=df['Cuisines'].value_counts()
print("cuisines distribution:")
print(cuisine_dis)

#Identify top cuisines & cities with Highest restaurants
top_cuisines=df['Cuisines'].value_counts().head(10)
print("Top cuisines are:")
print(top_cuisines)

top_cities=df['City'].value_counts().head(10)
print("cities with highest number of restaurants are:")
print(top_cities)

#Visualize the Restaurant Locations on map
plt.figure(figsize=(10,6))
plt.scatter(df['Longitude'],df['Latitude'],marker='o',color='blue',alpha=0.5)
plt.title('Restaurant Locations')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)
plt.show()

#Distribution of Restaurants across various cities
city_distribution=df['City'].value_counts().head(15)
plt.figure(figsize=(12,6))
city_distribution.plot(kind='bar')
plt.title('Distribution of restauarants Across cities')
plt.xlabel('City',fontsize=2)
plt.ylabel('NUmber of restaurants')
plt.xticks(rotation=45,ha='right')
plt.show()

#Distribution of Restaurants across various countries
city_distribution=df['Country Code'].value_counts()
plt.figure(figsize=(12,6))
city_distribution.plot(kind='bar')
plt.title('Distribution of restauarants Across countries')
plt.xlabel('Country Code',fontsize=10)
plt.ylabel('NUmber of restaurants')
plt.xticks(rotation=0,ha='right')
plt.show()



#Find correlation between Restaurant's location and it's rating
corr_lat_rating=df['Latitude'].corr(df['Aggregate rating'])
corr_lon_rating=df['Longitude'].corr(df['Aggregate rating'])
print("Correlation between Latitude and Rating:",corr_lat_rating)
print("Correlation between Longitude and Rating:",corr_lon_rating)

#Linear Regression model
df=pd.get_dummies(df,columns=['Has Table booking','Has Online delivery'],drop_first=True)
X=df[['Average Cost for two','Votes','Price range','Has Table booking_Yes','Has Online delivery_Yes']]
y=df['Aggregate rating']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("Mean Squared error:",mse)
print("R-squared:",r2)

plt.figure(figsize=(8,6))
plt.scatter(y_test,y_pred,color='blue')
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],linestyle='--',color='red')
plt.title('Actual vs predicted Ratings')
plt.xlabel('Actual ratings')
plt.ylabel('Predicted Ratings')
plt.grid(True)
plt.show()

#Decision Tree model
decision_tree_model=DecisionTreeRegressor(random_state=42)
decision_tree_model.fit(X_train,y_train)
y_pred2=decision_tree_model.predict(X_test)
mse_D_T=mean_squared_error(y_test,y_pred2)
r2_decision_tree=r2_score(y_test,y_pred2)
print("Mean Squared error2:",mse_D_T)
print("R-squared2:",r2_decision_tree)

#Random Forest model
random_forest_model=RandomForestRegressor(random_state=42)
random_forest_model.fit(X_train,y_train)
y_pred3=random_forest_model.predict(X_test)
mse_R_F=mean_squared_error(y_test,y_pred3)
r2_random_forest=r2_score(y_test,y_pred3)
print("Mean Squared error3:",mse_R_F)
print("R-squared3:",r2_random_forest)

#Relationship between cuisine type and rating
top_cuisines=df['Cuisines'].value_counts().head(15).index
df_t_c=df[df['Cuisines'].isin(top_cuisines)]
cuisine_ratings=df_t_c.groupby('Cuisines')['Aggregate rating'].mean().sort_values(ascending=False)
plt.figure(figsize=(12,6))
cuisine_ratings.plot(kind='bar')
plt.title('Mean Ratings by cuisine types')
plt.xlabel('Cuisine type')
plt.ylabel('Mean rating')
plt.xticks(rotation=45,ha='right')
plt.tight_layout()
plt.show()

#visualise distribution of ratings for each cuisine type
plt.figure(figsize=(12,6))
df_t_c.boxplot(column='Aggregate rating',by='Cuisines',figsize=(12,6),rot=45)
plt.title('Distribution of Ratings by cuisine types')
plt.xlabel('Cuisine type')
plt.ylabel('Rating')
plt.tight_layout()
plt.show()

#Most popular cuisines based on voting
cuisine_ratings=df_t_c.groupby('Cuisines')['Votes'].sum().sort_values(ascending=False)
print("Most popular cuisines based on votes")
print(cuisine_ratings.head(10))

#Find if any cuisines recieve highest rating

cuisine_ratings=df_t_c.groupby('Cuisines')['Aggregate rating'].mean().sort_values(ascending=False)
print("Cuisines with highest rating")
print(cuisine_ratings.head(10))

#Visualization of ratings over different charts
top_cuisines=df['Cuisines'].value_counts().head(15).index
df_top_cuisines=df[df['Cuisines'].isin(top_cuisines)]
#Distribution using Histogram
plt.figure(figsize=(10,6))
plt.hist(df['Aggregate rating'],bins=20,color='skyblue',edgecolor='black')
plt.title('Distribution of Ratings using Histogram')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()

#Distribution using Barplot for cuisine types
plt.figure(figsize=(12,6))
sns.barplot(x='Cuisines',y='Aggregate rating',data=df_top_cuisines,errorbar=None,palette='viridis')
plt.title('Mean ratings by Cuisines Type Using Barplot')
plt.xlabel('Cuisine Type')
plt.ylabel('Mean Rating')
plt.xticks(rotation=45,ha='right')
plt.tight_layout()
plt.show()

#Distribution using Boxplot for cuisine types
plt.figure(figsize=(12,6))
sns.barplot(x='Cuisines',y='Aggregate rating',data=df_top_cuisines,palette='Set2')
plt.title('Distribution of Ratings using Boxplot')
plt.xlabel('Cuisine Type')
plt.ylabel('Rating')
plt.xticks(rotation=45,ha='right')
plt.tight_layout()
plt.show()

#Compare average ratings oof different cuisines
sns.set(style="whitegrid")
cuisine_ratings=df_t_c.groupby('Cuisines')['Aggregate rating'].mean().sort_values(ascending=False)
plt.figure(figsize=(12,6))
sns.barplot(x=cuisine_ratings.index,y=cuisine_ratings.values,palette='viridis')
plt.title('Average Ratings by Cuisine Type ')
plt.xlabel('Cuisine type')
plt.ylabel('Aggregate rating')
plt.xticks(rotation=45,ha='right')
plt.tight_layout()
plt.show()

#Visualize the relationship between various features and target variable
#Scatter plot for numerical features
plt.figure(figsize=(12,6))
sns.scatterplot(x='Average Cost for two',y='Aggregate rating',data=df)
plt.title('Scatter plot of Feature1 vs Rating ')
plt.xlabel('Average Cost For Two')
plt.ylabel('Rating')
plt.show()

#Line plot for numerical features
plt.figure(figsize=(12,6))
sns.lineplot(x='Votes',y='Aggregate rating',data=df)
plt.title('Line plot of Feature2 vs Rating')
plt.xlabel('Votes')
plt.ylabel('Rating')
plt.show()

#Box plot for numerical features
plt.figure(figsize=(12,6))
sns.boxplot(x='Has Table booking_Yes',y='Aggregate rating',data=df)
plt.title('Box plot of Categorical Feature1 vs Rating ')
plt.xlabel('Has Table Booking')
plt.ylabel('Rating')
plt.show()

#Violon plot for numerical features
plt.figure(figsize=(12,6))
sns.violinplot(x='Has Online delivery_Yes',y='Aggregate rating',data=df)
plt.title('Violin plot of Categorical Feature2 vs Rating ')
plt.xlabel('Has Online Delivery')
plt.ylabel('Rating')
plt.show()

"""

#compare average ratings of restaurants with & without table booking
# Converting 'Yes'/'No' to boolean values 
df['Has Table booking'] = df['Has Table booking'].map({'Yes': True, 'No': False})
average_ratings_table_booking = df.groupby('Has Table booking')['Aggregate rating'].mean()
print("Average Ratings for Restaurants with and without Table Booking:")
print(average_ratings_table_booking)

#Availability of online delivery among restaurants with different price ranges
df[' Has Online delivery'] = df['Has Online delivery'].map({'Yes': True, 'No': False})
price_range_online_delivery = pd.crosstab(df['Price range'], df['Has Online delivery'])
price_range_online_delivery_percentage = price_range_online_delivery.div(price_range_online_delivery.sum(axis=1), axis=0) * 100
print("Cross-tabulation of Price Range and Online Delivery:")
print(price_range_online_delivery)

print("\nPercentage of Restaurants Offering Online Delivery within Each Price Range:")
print(price_range_online_delivery_percentage)

#Extracting the additional features 

# Add a column for the length of the restaurant name
df['Length of Restuarant Name'] = df['Restaurant Name'].apply(len)

# Add a column for the length of the address
df['Length of Address'] = df['Address'].apply(len)
print("DataFrame with Additional Features:")
print(df)


"""
# Count number of times each price range occurs
price_range_counts = df['Price range'].value_counts()
print("Price Range Counts:")
print(price_range_counts)
most_common_price_range = price_range_counts.idxmax()
print("\nMost Common Price Range:", most_common_price_range)


#Average rating for each price range
average_ratings = df.groupby('Price range')['Aggregate rating'].mean()
print("Average Ratings for Each Price Range:")
print(average_ratings)



# Identify the price range with the highest average rating
highest_avg_rating_price_range = average_ratings.idxmax()
print("\nPrice Range with Highest Average Rating:", highest_avg_rating_price_range)


# Define colors for each price range
price_range_colors = Rating color
print(price_range_colors)
    # Add more price ranges and their corresponding colors if needed

# Find the color corresponding to the price range with the highest average rating
highest_avg_rating_color = price_range_colors[highest_avg_rating_price_range]
print("Color representing the highest average rating:", highest_avg_rating_color)


# Optional: Visualize the average ratings with colors
plt.figure(figsize=(10, 6))
sns.barplot(x=average_ratings.index, y=average_ratings.values, palette=[price_range_colors[pr] for pr in average_ratings.index])
plt.title('Average Ratings by Price Range')
plt.xlabel('Price Range')
plt.ylabel('Average Rating')
plt.show()
"""
