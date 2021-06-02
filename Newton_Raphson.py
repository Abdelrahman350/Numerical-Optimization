import math

def func1(x):
    return x**2 - x - 1

def func2(x):
    return x**3 - 7*x**2 + 8*x -3

def func3(x):
    return x * math.cos(x) - x**2

def derivative(f, x):
    h = 0.000000000001
    return (f(x+h) - f(x-h))/(2*h)

def newtonRaphson(func, x0):
    N = 1000
    step = 0
    x = 0
    while abs(func(x0)) > 0.00000001:
        x = x0 - func(x0) / derivative(func, x0)
        x0 = x
        step = step + 1
        
        if step >= N:
            break
    return x
        

print(newtonRaphson(func=func1, x0=1))
print(newtonRaphson(func=func2, x0=5))
print(newtonRaphson(func=func3, x0=1))