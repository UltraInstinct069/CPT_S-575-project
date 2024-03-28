"""
Authors:    Abrar Akhyer Abir (abrarakhyer.abir@wsu.edu)
            Tashi Stirewalt (tashi.stirewalt@wsu.edu)
School:     Washington State University
Course:     CPT_S 575 - Data Science
Instructor: Dr. Assefaw Gebremedhin
Term:       Fall 2022
Date:       11/2022
Project:    Map Points Reduction
Tester for easy one off program end-to-end executions during code development.
"""

import pandas as pd
import map_points_reduction as mpr
import location_data_generator as ldg

# Location files initially given to us by T.A.
locations_df = pd.read_csv("Data/locations.csv")        # (638573, 3), 3 = ["datetime", "latitude", "longitude"]
locations5k_df = pd.read_csv("Data/locations.csv")    # (5000, 3), 3 = ["datetime", "latitude", "longitude"]

lat_max_range = locations5k_df["latitude"].max() - locations5k_df["latitude"].min()
lon_max_range = locations5k_df["longitude"].max() - locations5k_df["longitude"].min()

print("Maximum value ranges:\n\tLat = ", lat_max_range, "\n\tLon = ", lon_max_range)

# Grab mini-batch of samples from larger set
data_df = locations5k_df

# Create custom mini-batch of random ranged lat and lon values for easy case testing
# data_df = ldg.generateLocationData(number=100, min_lat=-5, max_lat=5, min_lon=-10, max_lon=10)

# Print out initial data before Map Points Reduction execution
print("INITIAL DATAFRAME:\n", data_df)

# Create instance of Map Points Reduction object with specified configuration (different config. = different results)
mpr1 = mpr.MPR(kilometers=True, grid_size=60, equatorial_basis='equatorial', polar_basis='polar')

# Print out configuration details for more specific information and for reference
mpr1.printConfig()

# Perform the Map Points Reduction process and specify whether to plot result for checking/demonstration
result = mpr1.mapPointReduce(data=data_df, plot=True)

# Print out data following process completion
print("MAP POINTS REDUCTION RESULT:\n", result)
