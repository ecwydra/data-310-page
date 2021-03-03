# Lab 3

## Question 1

An "ordinary least squares" (or OLS) model seeks to minimize the differences between your true and estimated dependent variable.

* **True -- This is the answer**
* False

## Question 2

Do you agree or disagree with the following statement:

In a linear regression model, all feature must correlate with the noise in order to obtain a good fit.

* Agree
* Disagree

Reasoning: I originally answered that I agreed with this statement because if the features don't correlate with noise, or outliers, then the model would get thrown off by outlier data and the fit wouldn't be as accurate.

## Question 3

* Write your own code to import L3Data.csv into python as a data frame.
* Then save the feature values  'days online','views','contributions','answers' into a matrix x and consider 'Grade' values as the dependent variable.
* If you separate the data into Train & Test with test_size=0.25 and random_state = 1234.
* If we use the features of x to build a multiple linear regression model for predicting y, then the root mean square error on the test data is close to:

1. 6.6523
2. 7.0123
3. 9.6312
4. **8.3244 -- This is the answer!**

```python
data = pd.read_csv('L3Data.csv')
data.head(3)

# data frame to matrix
feature_df = data[['days online', 'views', 'contributions', 'answers']]
target_df = data[['Grade']]

feature_matrix = feature_df.values
target_matrix = target_df.values

# separate into train and test
X_train, X_test, y_train, y_test = train_test_split(feature_matrix, target_matrix, test_size=0.25, random_state=1234)

# build multiple linear regression
scale = StandardScaler()
Xs_train = scale.fit_transform(X_train)
Xs_test  = scale.transform(X_test)
model = LinearRegression()
model.fit(Xs_train, y_train)
yhat = model.predict(Xs_test)

print(f'the mse for the test data is {np.sqrt(MSE(y_test, yhat))}')
```

## Question 4

In practice we determine the weights for linear regression with the "X_test" data.

* True
* **False -- I think it's this one. We determine the weights for linear regression with the train_test_split function, not with the var.**

## Question 5

Polynomial regression is best suited for functional relationships that are non-linear in weights.

* True
* False

## Question 6

Linear regression, multiple linear regression, and polynomial regression can be all fit using LinearRegression() from the sklearn.linear_model module in Python.

* **True -- This is the answer**
* False

## Question 7

* Write your own code to import L3Data.csv into python as a data frame.
* Then save the feature values  'days online','views','contributions','answers' into a matrix x and consider 'Grade' values as the dependent variable.
* If you separate the data into Train & Test with test_size=0.25 and random_state = 1234, then the number of observations we have in the Train data is

1. 22
2. **23 -- This is the answer**
3. 25
4. 24

```Python
# data frame
df = pd.read_csv('L3Data.csv')
df.head(3)

# features and targets saved
X = df[['days online', 'views', 'contributions', 'answers']]
y = df['Grade']

# separate into train and test
X_train, X_test, y_train, y_test = train_test_split(feature_matrix, target_matrix, test_size=0.25, random_state=1234)

X_train.shape
```

## Question 8

The gradient descent method does not need any hyper parameters.

* **True -- this is the answer!**
* False

## Question 9

To create and display a figure using matplotlib.pyplot that has visual elements (scatterplot, labeling of the axes, display of grid), in what order would the below code need to be executed?

1. import matplotlib.pyplot as plt

2. fig, ax = plt.subplots()

3. ax.scatter(X_test, y_test, color="black", label="Truth")
    ax.scatter(X_test, lin_reg.predict(X_test), color="green", label="Linear")
    ax.set_xlabel("Discussion Contributions")
    ax.set_ylabel("Grade")

4. ax.grid(b=True,which='major', color ='grey', linestyle='-', alpha=0.8)
    ax.grid(b=True,which='minor', color ='grey', linestyle='--', alpha=0.2)
    ax.minorticks_on()

## Question 10

Which of the following is nonlinear?

The one that has weights the aren't just beta, there are exponents and stuff attached to the weights.
