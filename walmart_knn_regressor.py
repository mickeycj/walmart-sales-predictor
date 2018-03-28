import pandas as pd
from datetime import datetime
from sklearn.neighbors import KNeighborsRegressor

def transform_and_normalize_date(X):
    max_day = 0
    for i in range(int(X.size/4)):
        X[i][2] = datetime.strptime(X[i][2], '%Y-%m-%d').timetuple().tm_yday
        if max_day < X[i][2]: max_day = X[i][2]
    for i in range(int(X.size/4)):
        X[i][2] /= max_day

train_df = pd.read_csv('train.csv')
X = train_df[['Store', 'Dept', 'Date', 'IsHoliday']].as_matrix()
y = train_df[['Weekly_Sales']].as_matrix()
transform_and_normalize_date(X)

knn_regressor = KNeighborsRegressor(n_neighbors=50, weights='distance', p=1)
knn_regressor.fit(X, y)

test_df = pd.read_csv('test.csv')
T = test_df[['Store', 'Dept', 'Date', 'IsHoliday']].as_matrix()
ids = test_df[['Store', 'Dept', 'Date']].as_matrix()
transform_and_normalize_date(T)

with open('submission.csv', 'w') as file:
    file.write('Id,Weekly_Sales\n')
    for i in range(int(T.size/4)):
        (store, dept, date, is_holiday) = (T[i][0], T[i][1], T[i][2], T[i][3])
        file.write("%s_%s_%s," % (ids[i][0], ids[i][1], ids[i][2]))
        file.write(str(knn_regressor.predict([[store, dept, date, is_holiday]])[0][0]))
        file.write('\n')