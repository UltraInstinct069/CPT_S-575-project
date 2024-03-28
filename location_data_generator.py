"""
Authors:    Abrar Akhyer Abir (abrarakhyer.abir@wsu.edu)
            Tashi Stirewalt (tashi.stirewalt@wsu.edu)
School:     Washington State University
Course:     CPT_S 575 - Data Science
Instructor: Dr. Assefaw Gebremedhin
Term:       Fall 2022
Date:       11/2022
Project:    Map Points Reduction
"""

import numpy as np
import pandas as pd


def generateLocationData(number: int = 100, min_lat: float = -90, max_lat: float = 90, min_lon: float = -180,
                         max_lon: float = 180):
    """
    Creates a pandas Dataframe of columns "latitude" and "longitude" with random values uniformly from [min, max) of
    passed lat and lon parameter values.

    :param number: integer number of samples to generate.
    :param min_lat: minimum value for latitude values to take. Must be less than or equal to max_lat value and greater
    than or equal to -90.
    :param max_lat: maximum value for latitude values to take. Must be greater than or equal to min_lat value and less
    than or equal to 90.
    :param min_lon: minimum value for longitude values to take. Must be less than or equal to max_lat value and greater
    than or equal to -180.
    :param max_lon: maximum value for longitude values to take. Must be greater than or equal to min_lat value and less
    than or equal to 90.
    :return: pandas Dataframe of columns "latitude" and "longitude" with random values uniformly from [min, max) of
    passed lat and lon parameter values.
    :raises ValueError: raises exception when an invalid parameter value is passed
    """
    if number <= 0:
        raise ValueError("Input number of '", number, "' is non-positive. Must be greater than 0!")
    if min_lat < -90:
        raise ValueError("Input min_lat of '", min_lat, "' is out of bounds. Must be greater than or equal to -90!")
    if max_lat > 90:
        raise ValueError("Input max_lat of '", max_lat, "' is out of bounds. Must be less than or equal to 90!")
    if min_lat > max_lat:
        raise ValueError("Input min_lat of '", min_lat, "' is greater than input max_lat of '", max_lat,
                         "'!. min_lat must be less than or equal to max_lat!")
    if min_lon < -180:
        raise ValueError("Input min_lon of '", min_lon, "' is out of bounds. Must be greater than or equal to -180!")
    if max_lon > 180:
        raise ValueError("Input max_lon of '", max_lon, "' is out of bounds. Must be less than or equal to 180!")
    if min_lon > max_lon:
        raise ValueError("Input min_lon of '", min_lon, "' is greater than input max_lon of '", max_lon,
                         "'!. min_lon must be less than or equal to max_lon!")
    lat_values = np.random.uniform(low=min_lat, high=max_lat, size=number).reshape(-1, 1)
    lon_values = np.random.uniform(low=min_lon, high=max_lon, size=number).reshape(-1, 1)
    return pd.DataFrame(np.concatenate((lat_values, lon_values), axis=1), columns=["latitude", "longitude"])
