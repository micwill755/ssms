# Function vs Signal:

## Function = A mathematical rule or formula

- sin(x) is a function - it's the rule "given any input x, calculate the sine"
- You can evaluate it at any point: sin(0), sin(π/2), sin(2.7), etc.

## Signal = A sequence of actual values over time

- [0.1, 0.5, -0.2, 0.8, ...] - this is a signal
- It's the actual data you collect or measure
- Like audio samples, stock prices, or temperature readings

You create a signal BY using a function. You take the sin() function and evaluate it at specific time points (0, 1, 2, 3...) to get an array of numbers. That array IS your signal.

when we say "discrete signal," we mean the array of numbers we got by sampling a continuous function at discrete time points.

# General Function Formulas

cos(x) = sin(x + π/2)
sin(x) = cos(x - π/2)