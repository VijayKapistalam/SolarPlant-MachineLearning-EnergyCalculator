import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from scipy import stats

#Read the CSV files into seperate data frames
generation_data = pd.read_csv("Plant_1_Generation_Data.csv")
weather_data = pd.read_csv("Plant_1_Weather_Sensor_Data.csv")

#Remove additional characters from DATE_TIME column in generation_data
generation_data['DATE_TIME'] = generation_data['DATE_TIME'].str.replace(r'\s+\d+:\d+', '')

#Fixing date for Generation_Data
generation_data['DATE_TIME'] = pd.to_datetime(generation_data['DATE_TIME'], format='mixed')

#Fixing date for Generation_Data
generation_data['DATE_TIME'] = pd.to_datetime(generation_data['DATE_TIME'], dayfirst=True)
weather_data['DATE_TIME'] = pd.to_datetime(weather_data['DATE_TIME'])

#Concactenate the DataFrames using the 'concat()' function
combined_data = pd.merge(generation_data, weather_data, on=['DATE_TIME', 'PLANT_ID'])

#Cleaning Data
# print(combined_data1.isnull())
# print(combined_data2.isnull())
# print(combined_data1.duplicated().any())
# print(combined_data2.duplicated().any())

#Feature Engineering
    #Normalize Features
normalized_data = combined_data.copy()

    #Turning combined_data into z-scores(standardizing data) so they are on similar scale
normalized_data[['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']] = \
    (combined_data[['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']] -
     combined_data[['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']].mean()) / \
    combined_data[['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']].std()


    #Capture time-based features, which can be useful in capturing temporal patterns.
normalized_data['HOUR'] = combined_data['DATE_TIME'].dt.hour
normalized_data['MONTH'] = combined_data['DATE_TIME'].dt.month


    #Calculate aggregate statistics, for data grouped by 'DATE_TIME'
normalized_data['DC_POWER_MEAN'] = combined_data.groupby('DATE_TIME')['DC_POWER'].transform('mean')
normalized_data['AC_POWER_MEAN'] = combined_data.groupby('DATE_TIME')['AC_POWER'].transform('mean')
normalized_data['DAILY_YIELD_SUM'] = combined_data.groupby('DATE_TIME')['DAILY_YIELD'].transform('sum')
normalized_data['TOTAL_YIELD_SUM'] = combined_data.groupby('DATE_TIME')['TOTAL_YIELD'].transform('sum')


    #Additional Features
normalized_data['TEMPERATURE_DIFF'] = combined_data['MODULE_TEMPERATURE'] - combined_data['AMBIENT_TEMPERATURE']
normalized_data['IRRADIATION_SQUARED'] = combined_data['IRRADIATION']**2

    #New dataframe with selected features
selected_features = normalized_data[['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION', 'TEMPERATURE_DIFF', 'IRRADIATION_SQUARED']]


#Check correlation coefficients
    #FEATURE SELECTION MAY HAVE TO BE DIFFERENT FOR EACH DATA SET -- (AND MIGHT HAVE TO CHANGE TOTAL_YIELD_SUM TO JUST TOTAL_YIELD)**DON'T KNOW IF THIS ONE NECESSARILY IMPORTANT
# correlation1 = selected_features1.corr()['DC_POWER_MEAN']
# correlation2 = selected_features2.corr()['DC_POWER_MEAN']
# print(correlation2)
# print()
# print(correlation1)
# print()
# print()


#Split the data into a training and testing set for selected_features
X_train1, X_test1, y_train1, y_test1 = train_test_split(selected_features, normalized_data['DC_POWER_MEAN'], test_size=0.2, random_state=42)

#Train a Random Forest Regression model for selected_features
regressor = RandomForestRegressor()
regressor.fit(X_train1, y_train1)

#Make predictions on the testing set for selected_features
y_pred1 = regressor.predict(X_test1)

#Feature importance
# feature_importance = regressor.feature_importances_
#     #DataFrame to store feature importances
# importance_df = pd.DataFrame({'Feature': selected_features.columns, 'Importance': feature_importance})
#         #Sort dataFrames
# importance_df = importance_df.sort_values(by='Importance', ascending=False)
#     #Print feature importances
# print(importance_df)


#Evaluate the model
mse = mean_squared_error(y_test1, y_pred1)
r2 = r2_score(y_test1, y_pred1)

print("Model Performance for the historical solar plant")
print("Mean Squared Error:", mse)
print("R2 Score:", r2)
print()



# #User Data Input

print("Enter the file path of CSV file containing your plant's weather data.")
csv_path_weather = input("This file should include your plant's Ambient Temperature, Module Temperature, and Irradiation values: ")
print()
user_data = pd.read_csv(csv_path_weather)

#Data verification
expected_columns = ['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']
if not set(expected_columns).issubset(user_data.columns):
    print("Error: The CSV file does not contain the required columns.")
    exit(1)

#cleaning user data
user_data.dropna(inplace=True)

z_scores = user_data[['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']].apply(stats.zscore)
user_data = user_data[(z_scores < 3).all(axis=1)]

user_data['AMBIENT_TEMPERATURE'] = user_data['AMBIENT_TEMPERATURE'].astype(float)
user_data['MODULE_TEMPERATURE'] = user_data['MODULE_TEMPERATURE'].astype(float)
user_data['IRRADIATION'] = user_data['IRRADIATION'].astype(float)


#normalize user data
normalized_userData = user_data.copy()
normalized_userData = normalized_userData[['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']] = \
                        (user_data[['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']] - \
                         user_data[['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']].mean()) / \
                        user_data[['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']].std()

#user selected features
normalized_userData['TEMPERATURE_DIFF'] = user_data['MODULE_TEMPERATURE'] - user_data['AMBIENT_TEMPERATURE']
normalized_userData['IRRADIATION_SQUARED'] = user_data['IRRADIATION']**2

#New dataframe with selected features
selected_userFeatures = normalized_userData[['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION', 'TEMPERATURE_DIFF', 'IRRADIATION_SQUARED']]

 #Make predictions on the testing set for selected_features
user_pred = regressor.predict(selected_userFeatures)

printHead = ""
printHead = input("If you'd like to see all your data, enter 'yes' (or press Enter for summary): ")
if(printHead == 'yes'):
    # Concatenate predicted DC_POWER_MEAN to the user data DataFrame
    user_data['PREDICTED_DC_POWER_MEAN'] = user_pred

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    # Display user data along with the predicted DC_POWER_MEAN
    print("Here is your predicted Average DC Power: ")
    print(user_data['PREDICTED_DC_POWER_MEAN'])

    # Reset the display options to the default values
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
else:
    user_data['PREDICTED_DC_POWER_MEAN'] = user_pred
    print("Here is your predicted Average DC Power: ")
    print(user_data['PREDICTED_DC_POWER_MEAN'])
