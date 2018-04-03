import pandas as pd
from datetime import datetime
from sklearn.neighbors import KNeighborsRegressor

def transform_and_normalize_data(df, stores_df):
    df = pd.merge(df, stores_df, on='Store')
    X = df[['Store', 'Type', 'Size', 'Dept', 'Date', 'IsHoliday']].as_matrix()

    max_size = 0
    max_week = 0
    for i in range(int(X.size/6)):
        X[i][1] = ord(X[i][1]) - ord('A')
        if max_size < X[i][2]: max_size = X[i][2]
        X[i][4] = datetime.strptime(X[i][4], '%Y-%m-%d').isocalendar()[1]
        if max_week < X[i][4]: max_week = X[i][4]
    for i in range(int(X.size/6)):
        X[i][2] /= max_size
        X[i][4] /= max_week

    return X

# Retrieve stores' information.
stores_df = pd.read_csv('stores.csv')

# Retrieve and pre-process the training set.
train_df = pd.read_csv('train.csv')
X = transform_and_normalize_data(train_df, stores_df)
y = train_df[['Weekly_Sales']].as_matrix()

# Train the kNN model.
knn_regressor = KNeighborsRegressor(n_neighbors=50, weights='distance', p=1)
knn_regressor.fit(X, y)

# Retrieve and pre-process the testing set.
test_df = pd.read_csv('test.csv')
T = transform_and_normalize_data(test_df, stores_df)
ids = test_df[['Store', 'Dept', 'Date']].as_matrix()

# Write the prediction results to submission.csv file.
with open('submission.csv', 'w') as file:
    file.write('Id,Weekly_Sales\n')
    for i in range(int(T.size/6)):
        (store, store_type, size, dept, date, is_holiday) = (T[i][0], T[i][1], T[i][2], T[i][3], T[i][4], T[i][5])
        file.write("%s_%s_%s," % (ids[i][0], ids[i][1], ids[i][2]))
        file.write(str(knn_regressor.predict([[store, store_type, size, dept, date, is_holiday]])[0][0]))
        file.write('\n')
