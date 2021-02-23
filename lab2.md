# Lab 2

## Question 1

Regardless whether we know or not the shape of the distribution of a random variable, an interval centered around the mean whose total length is 8 standard deviations is guaranteed to include at least a certain percantage of data. This guaranteed minimal value as a percentage is **93.75%**. I originally said the answer was 93%, but should have been more specific about the percentage.

```python
std = 8

answer = 1 - (1 / (std ** std))
answer * 100 
```

## Question 2

For scaling data by quantiles we need to compute the z-scores first.

* True
* False -- This is the answer

## Question 3

In the 'mtcars' dataset the zscore of an 18.1mpg car is -0.335572. I originally said the answer was -0.33, but again should have been more specific and included more significant figures.

```python
data = pd.read_csv('mtcars.csv')
data_copy = data

# get the mean for mpg
mean = data['mpg'].mean()

# get the standard deviation
std = data['mpg'].std()

# get specific car (x)
data_filtered_idx = (data['mpg'] == 18.1)
data_filtered = data[data_filtered_idx]
x = 18.1

# compute z score
result = (x - mean) / std
```

## Question 4

In the 'mtcars' dataset determine the percentile of a car that weighs 3520bs is 68.75%. 

```python
# how many total wts are there
num_weight = data.wt.count()
# how many wts are equal to 3520 or lower
data_weight_idx = (data['wt'] <= 3.520)
data_weight = data[data_weight_idx]
num_below = data_weight.wt.count()
# divide
(num_below / num_weight) * 100
```

## Question 5

A finite sum of squared quantities that depends on some parameters (weights), always has a minimum value.

* True -- This is the answer
* False

## Question 6

For the 'mtcars' data set use a linear model to predict the mileage of a car whose weight is 2800lbs. The answer with only the first two decimal places and no rounding is:

```python
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler, QuantileTransformer

qtn = QuantileTransformer(n_quantiles=100)
qtn.fit_transform(data[['wt']])*100
np.argmax(qtn.fit_transform(data[['wt']])*100)

v = data[['wt']].values
v[np.argmin(qtn.fit_transform(data[['wt']])*100)]

x = data[['wt']]
y = data[['mpg']]
lm = linear_model.LinearRegression()
model = lm.fit(x,y)

lm.predict([[2.8]])
```

## Question 7

If, for running the gradient descent algorithm, you consider the learning_rate = 0.01, the number of iterations = 10000 and the initial slope and intercept equal to 0, then the optimal value of the sum of the squared residuals is 278.2219. When answering this question, I accidentally put the minimized cost when I should have put the last squared residual that I printed (that number was 278.3219375435502).

```python
def compute_cost(b, m, data):
    total_cost = 0
    
    # number of datapoints in training data
    N = float(len(data))
    
    # Compute sum of squared errors
    for i in range(0, len(data)):
        x = data[i, 0]
        y = data[i, 1]
        total_cost += (y - (m * x + b)) ** 2
        
    print(total_cost)
        
    # Return average of squared error
    return total_cost/(2*N)
```

I found the answer by adding a print statement to the compute_cost method. The print statement prints out the last squared residual computation, which matches the answer.

## Question 8

If we have one input variable and one output, the process of determining the line of best fit may not require the calculation of the intercept inside the gradient descent algorithm.

* True -- This is maybe the answer
* False

## Question 9

For the line of regression in the case of the example we discussed with the 'mtcars' data set the meaning of the intercept is...

This slope has **no interpretable meaning**. It's the slope of the line that has meaning, not the intercept.

## Question 10

The slope of the regression line always remains the same if we scale the data by z-scores.

* True -- This is the answer
* False
