import numpy as np 
import pandas as pd 
import os
import pickle

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn import linear_model
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


file = open('/kaggle/input/cs-4780-covid-case-hunters/covid_dataset.pkl', 'rb')
checkpoint = pickle.load(file)
file.close()
X_train, y_train, X_val, y_val, X_test = checkpoint["X_train"], checkpoint["y_train_log_pos_cases"], checkpoint["X_val"], checkpoint["y_val_log_pos_cases"], checkpoint["X_test"]



enc = OneHotEncoder()

X_combined = np.concatenate((X_val, X_train, X_test), axis=0)
X_combined = np.reshape(np.where(X_combined[:,0] != None, X_combined[:,0], 11), (-1,1))
enc = enc.fit(X_combined)
X_train_cat = np.reshape(np.where(X_train[:,0] != None, X_train[:,0], 11), (-1,1))
X_train_cat = enc.transform(X_train_cat).toarray()
X_val_cat = np.reshape(np.where(X_val[:,0] != None, X_val[:,0], 11), (-1,1))
X_val_cat = enc.transform(X_val_cat).toarray()
X_test_cat = np.reshape(np.where(X_test[:,0] != None, X_test[:,0], 11), (-1,1))
X_test_cat = enc.transform(X_test_cat).toarray()
X_train = np.concatenate((X_train[:,0:],X_train_cat), axis=1)
X_val = np.concatenate((X_val[:,0:],X_val_cat), axis=1)
X_test = np.concatenate((X_test[:,0:],X_test_cat), axis=1)



imp3 = SimpleImputer(missing_values=np.nan)


imp3 = imp3.fit(np.concatenate((X_train, X_val, X_test), axis=0))

X_train_imp = imp3.transform(X_train)

X_val_imp = imp3.transform(X_val)

X = np.concatenate((X_val_imp,X_train_imp),axis=0)
y = np.concatenate((y_val,y_train))

X_test_imp = imp3.transform(X_test)


scaler = StandardScaler()
scaler.fit(np.concatenate((X,X_test_imp), axis=0)) 
X = scaler.transform(X)

X_train_imp = scaler.transform(X_train_imp)
X_val_imp = scaler.transform(X_val_imp)
X_test_imp = scaler.transform(X_test_imp)



#pca = PCA(n_components=22)
#X = pca.fit_transform(X)

thing = .85*X.shape[0]
real_X_val = X[int(thing):]
real_y_val = y[int(thing):]
X = X[:int(thing)]
y = y[:int(thing)]

hard_data = []


Q = int(thing)
kf = KFold(n_splits=Q)
sum = 0
loss = 0
for train_index, test_index in kf.split(X):
    vc_X_train, vc_X_val = X[train_index], X[test_index]
    vc_y_train, vc_y_val = y[train_index], y[test_index]

    #####
    model = KernelRidge(kernel='rbf', gamma=.0055, alpha=.03)
    #####
    model.fit(vc_X_train, vc_y_train)
    predictions = model.predict(vc_X_val)
    if mean_squared_error(vc_y_val, predictions) > 4.5:
        print(test_index)
        print(mean_squared_error(vc_y_val, predictions))
        hard_data.append(test_index)
    sum = sum + mean_squared_error(vc_y_val, predictions)
    predictions = model.predict(X_train_imp)
    loss = loss + mean_squared_error(y_train, predictions)



challenges = X[hard_data[0],:]
challenges_y = y[hard_data[0]]

for double in range(2):
    for num, i in enumerate(hard_data):
        if num!=0:
            challenges = np.concatenate((challenges, X[i,:]), axis=0)
            challenges_y = np.concatenate((challenges_y, y[i]))


challenges = np.concatenate((challenges, X), axis=0)
challenges_y = np.concatenate((challenges_y, y))
challenges_y = np.reshape(challenges_y, (np.size(challenges_y),1))
combined = np.concatenate((challenges, challenges_y), axis=1)
rng = np.random.default_rng()
rng.shuffle(combined, axis=0)

challenges = combined[:,:-1]
challenges_y = np.reshape(combined[:,-1], (np.size(challenges_y,)))


model = GradientBoostingRegressor(max_depth=4, n_estimators=8000, learning_rate=.05, n_iter_no_change=200)
model.fit(challenges, challenges_y)
test_pred = model.predict(X_test_imp)

pd.DataFrame(test_pred).to_csv("submission.csv", header=["cases"], index_label="id")