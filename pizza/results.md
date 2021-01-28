# Run 1

## Inputs

```
lr = 0.01, iterations = 10000
```

## Outputs

```
Iteration 1551 => Loss: 22.863567

w=1.100, b=12.930
Prediction: x=20 => y=34.93
```

# Run 2

## Inputs

```
lr = 0.001, iterations = 10000
```

## Outputs

```
Iteration 9999 => Loss: 30.807105
Traceback (most recent call last):
  File "linear_regression.py", line 63, in <module>
    w, b = train(X, Y, iterations=10000, lr=0.001)
  File "linear_regression.py", line 56, in train
    raise Exception("Couldn't converge within %d iterations" % iterations)
Exception: Couldn't converge within 10000 iterations
```

# Run 3

## Inputs

```
lr = 0.1, iterations = 10000
```

## Outputs

```
Iteration  141 => Loss: 23.668667

w=1.200, b=11.700
Prediction: x=20 => y=35.70
```

# Run 4

## Inputs

```
lr = 0.001, iterations = 20000
```

## Outputs

```
Iteration 15767 => Loss: 22.842783

w=1.082, b=13.161
Prediction: x=20 => y=34.80
```
