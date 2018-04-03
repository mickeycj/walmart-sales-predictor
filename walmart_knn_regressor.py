import pandas as pd
from datetime import datetime
from sklearn.neighbors import KNeighborsRegressor

def transform_and_normalize_data(df):
    X = df[['Store', 'Dept', 'Date', 'IsHoliday']].as_matrix()

    max_week = 0
    for i in range(int(X.size/4)):
        X[i][2] = datetime.strptime(X[i][2], '%Y-%m-%d').isocalendar()[1]
        if max_week < X[i][2]: max_week = X[i][2]
    for i in range(int(X.size/4)):
        X[i][2] /= max_week

    return X

# Retrieve and pre-process the training set.
train_df = pd.read_csv('train.csv')
X = transform_and_normalize_data(train_df)
y = train_df[['Weekly_Sales']].as_matrix()

# Train the kNN model.
knn_regressor = KNeighborsRegressor(n_neighbors=50, weights='distance', p=1)
knn_regressor.fit(X, y)

# Retrieve and pre-process the testing set.
test_df = pd.read_csv('test.csv')
T = transform_and_normalize_data(test_df)
ids = test_df[['Store', 'Dept', 'Date']].as_matrix()

# Write the prediction results to submission.csv file.
with open('submission.csv', 'w') as file:
    file.write('Id,Weekly_Sales\n')
    for i in range(int(T.size/4)):
        (store, dept, date, is_holiday) = (T[i][0], T[i][1], T[i][2], T[i][3])
        file.write("%s_%s_%s," % (ids[i][0], ids[i][1], ids[i][2]))
        file.write(str(knn_regressor.predict([[store, dept, date, is_holiday]])[0][0]))
        file.write('\n')
