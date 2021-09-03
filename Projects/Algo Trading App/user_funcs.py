# user_funcs.py
"""
user-defined functions for the Streamlit mohr's circle app
"""

import numpy as np


def mohr_c(stress_x, stress_y, shear):
    """
    input arguments:
        stress_x: int or float
        stress_y: int or float
        shear: int or float

    mohr_c() outputs two values for the circle, center, and radius

    output:
        C, R
        C is x-value of the circle center
        R is the radius of the circle
    """
    C = (stress_x + stress_y) / 2
    R = ((((stress_x - stress_y) / 2) ** 2) + shear ** 2) ** 0.5
    return C, R


def c_array(C, R, n=100):
    """
    input arguments:
        C: int or float, the x-value of a circle center
        R: int or float, the radius of the circle
        n: int, number of points (optional)

    c_array() outputs two NumPy arrays of x and y values that creates the circle

    output:
        x, y
        x is a NumPy array with n-values that contains the x-values to build the circle
        y is a NumPy array with n-values that contains the y-value to build the circle
    """
    t = np.linspace(0, 2 * np.pi, n + 1)
    x = R * np.cos(t) + C
    y = R * np.sin(t)
    return x, y


def X_Y(stress_x, stress_y, shear):
    """build the arrays that describe the line X-Y between the known points"""
    X = np.array([stress_x, stress_y])
    Y = np.array([-shear, shear])
    return X, Y