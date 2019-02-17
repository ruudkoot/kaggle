import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

iowa_file_path = 'data/train.csv'
test_data_path = 'data/test.csv'

home_data = pd.read_csv(iowa_file_path)
test_data = pd.read_csv(test_data_path)

# one-hot encoding
home_data = pd.get_dummies(home_data)
test_data = pd.get_dummies(test_data)

home_y = home_data.SalePrice
home_X = home_data.drop(['SalePrice'], axis=1)
home_X, test_X = home_X.align(test_data, join='inner', axis=1)

# imputation
home_X_copy = home_X.copy()
test_X_copy = test_X.copy()
cols_with_missing = [col for col in home_X_copy.columns if home_X_copy[col].isnull().any()]
for col in cols_with_missing:
        home_X_copy[col + '_was_missing'] = home_X_copy[col].isnull()
        test_X_copy[col + '_was_missing'] = test_X_copy[col].isnull()
my_imputer = SimpleImputer()
home_X = my_imputer.fit_transform(home_X_copy)
test_X = my_imputer.fit_transform(test_X_copy)

rf_model_on_full_data = RandomForestRegressor(n_estimators=924)
rf_model_on_full_data.fit(home_X,home_y)

# make predictions which we will submit.
test_preds = rf_model_on_full_data.predict(test_X)

output = pd.DataFrame({'Id': test_data.Id, 'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)

print("Done!")

#########################

train_X, val_X, train_y, val_y = train_test_split(home_X, home_y, random_state=1)

def tune(n_est):
    rf_model = RandomForestRegressor(n_estimators=n_est,random_state=1)
    rf_model.fit(train_X, train_y)
    rf_val_predictions = rf_model.predict(val_X)
    rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)
    return rf_val_mae

min_n_est,min_mae=None,None
for n_est in range(1,10000):
    mae=tune(n_est)
    if min_mae == None or mae < min_mae:
        min_n_est,min_mae = n_est,mae
        print(min_n_est,min_mae)
print(min_n_est,min_mae)
rf_val_mae=min_mae

print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))
