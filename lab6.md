# Lab 6

## Multiple Choice Questions

1. MC1 True
2. MC2 True
3. MC3 Number of trees in the forest
5. MC5 Classified as malignant, but really was benign
6. MC6 True
7. MC7 A node which contains all data points
8. MC8 Probability based classification
9. MC9 Degree of class imbalance
10. MC10 The input variables ("Dendrites")

## Practical Questions

### Data Preprocessing

```python
data = load_breast_cancer()
X = data.data
X_names = data.feature_names
y = data.target

df = pd.DataFrame(X, columns=X_names)

features = ['mean radius', 'mean texture']
Xf = df[features].values
yf = data.target

ss = StandardScaler()
Xs = ss.fit_transform(Xf)
```

### Question 4
```python
# classification tree
dt_class = tree.DecisionTreeClassifier(random_state=1693, max_depth=5)
dt_class.fit(X,y)
y_pred = dt_class.predict(X)
dt_cm = confusion_matrix(y, y_pred)
# naive bayes
nb_class = GaussianNB()
nb_class.fit(X,y)
y_pred = nb_class.predict(X)
cm = confusion_matrix(y, y_pred)
# random forest
rf_class = RandomForestClassifier(random_state=1693, max_depth=5, n_estimators = 1000)
rf_class.fit(X,y)
y_pred = rf_class.predict(X)
rf_cm = confusion_matrix(y,y_pred)
```

Classification Tree: 0 FN
Naive Bayes: 10 FN
Random Forest: 0 FN

### Question 14 and 15

```python
def validation(X,y,k,model):
    PA_IV = []
    PA_EV = []
    pipe = Pipeline([('scale',scale),('Classifier',model)])
    kf = KFold(n_splits=k,shuffle=True,random_state=1693)
    for idxtrain, idxtest in kf.split(X):
        X_train = X[idxtrain,:]
        y_train = y[idxtrain]
        X_test = X[idxtest,:]
        y_test = y[idxtest]
        pipe.fit(X_train,y_train)
        PA_IV.append(accuracy_score(y_train,pipe.predict(X_train)))
        PA_EV.append(accuracy_score(y_test,pipe.predict(X_test)))
    return np.mean(PA_IV), np.mean(PA_EV)

model = GaussianNB()
scale = StandardScaler()
validation(Xf,yf,10,model) # -> 0.8805137844611528

model = RandomForestClassifier(random_state=1693, max_depth=7, n_estimators = 100)
scale = StandardScaler()
validation(Xf,yf,10,model) # -> 0.8822368421052632
```

### Question 16

```python
# create model
model = Sequential()

model.add(Dense(16, kernel_initializer='random_normal', input_dim=2, activation='relu'))
model.add(Dense(8, kernel_initializer='random_normal', input_dim=2, activation='relu'))
model.add(Dense(4, kernel_initializer='random_normal', input_dim=2, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics= 'accuracy')
# fit the model
model.fit(Xs, yf, epochs=150, verbose=0, validation_split=0.25, batch_size=10, shuffle=False)
# find accuracy
_, accuracy = model.evaluate(Xs, yf) # -> Accuracy on the Test Data: 88.92794251441956
```
