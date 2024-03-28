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

import math  # for math.pi use in _getDeltas()
import matplotlib.cm as cm  # for distinct cell coloring in _createPlot()
import matplotlib.pyplot as plt  # for scatter plotting in _createPlot()
import numpy as np  # for vectorized arithmetic operations
import pandas as pd  # for tabular form with column names and easier data viewing

# Global constants for different Earth Radius bases
# SOURCE: https://en.wikipedia.org/wiki/Earth_radius
EQUATORIAL_RADIUS_KM = 6378     # Earth's equatorial radius in kilometers
EQUATORIAL_RADIUS_M = 3963      # Earth's equatorial radius in miles
POLAR_RADIUS_KM = 6357          # Earth's polar radius in kilometers
POLAR_RADIUS_M = 3950           # Earth's polar radius in miles
AVERAGE_RADIUS_KM = 6371        # Earth's global average radius in kilometers
AVERAGE_RADIUS_M = 3959         # Earth's global average radius in miles


class MPR:  # Map Points Reduce (MPR)
    """ Class for Map Points Reduction process for "latitude" and "longitude" panda Dataframe's """

    def __init__(self, kilometers: bool = True, grid_size: float = 25, equatorial_basis: str = 'equatorial',
                 polar_basis: str = 'polar'):
        """
        Constructor for Map Points Reduction (MPR) instance to maintain parameter configuration between multiple
        mapping processes in case mini-batching large datasets is necessary.

        :param kilometers: grid size by unit type where kilometers = True is km and kilometers = False is miles.
        :param grid_size: float for grid size in kilometers/miles as set by parameter 'kilometers'.
        :param equatorial_basis: string (cap sensitive) 'equatorial'/'polar'/'average' for Earth Radius distance to use
            for longitude delta approximation.
        :param polar_basis: string (cap sensitive) 'equatorial'/'polar'/'average' for Earth Radius distance to use for
            latitude delta approximation.
        :raises ValueError: raises exception when an invalid parameter value is found during parameter checking.
        """
        # Initialize class attribute values for reference to instance configuration details via printConfig()
        self.kilometers = kilometers
        self.grid_size = grid_size
        self.equatorial_basis = equatorial_basis
        self.polar_basis = polar_basis

        # declared and init to 0, to be set inside _checkParams()
        self.equatorial_radius = 0
        self.polar_radius = 0

        # Checks parameter values are valid and initializes corresponding radius equatorial and polar attribute values
        self._checkParams()

        # Compute deltas based on current configuration and saves values as class attributes for mapPointReduce()
        self.delta_lon = 0
        self.delta_lat = 0
        self._getDeltas()

    def _checkParams(self):
        """
        Private function for checking parameters passed into the class constructor and for setting class attributes to
        global constant Earth radius values that correspond to the differing bases specified by the passed parameters.
        Will throw a ValueError Exception in any case where a passed parameter value is found to be invalid.

        :raises ValueError: raises exception when an invalid parameter value is found.
        """
        if self.kilometers:  # CASE 1: kilometers = TRUE, use kilometers
            if self.equatorial_basis == 'equatorial':  # Equatorial
                self.equatorial_radius = EQUATORIAL_RADIUS_KM
            elif self.equatorial_basis == 'average':
                self.equatorial_radius = AVERAGE_RADIUS_KM
            elif self.equatorial_basis == 'polar':
                self.equatorial_radius = POLAR_RADIUS_KM
            else:
                raise ValueError("Input equatorial_basis of '", self.equatorial_basis,
                                 "' is INVALID! Valid options include 'equatorial', 'polar', or, 'average'.")
            if self.polar_basis == 'polar':  # Polar
                self.polar_radius = POLAR_RADIUS_KM
            elif self.polar_basis == 'average':
                self.polar_radius = AVERAGE_RADIUS_KM
            elif self.polar_basis == 'equatorial':
                self.polar_radius = EQUATORIAL_RADIUS_KM
            else:
                raise ValueError("Input polar_basis of '", self.polar_basis,
                                 "' is INVALID! Valid options include 'equatorial', 'polar', or, 'average'.")
        else:  # CASE 2: kilometers = FALSE, use miles
            if self.equatorial_basis == 'equatorial':  # Equatorial
                self.equatorial_radius = EQUATORIAL_RADIUS_M
            elif self.equatorial_basis == 'average':
                self.equatorial_radius = AVERAGE_RADIUS_M
            elif self.equatorial_basis == 'polar':
                self.equatorial_radius = POLAR_RADIUS_M
            else:
                raise ValueError("Input equatorial_basis of '", self.equatorial_basis,
                                 "' is INVALID! Valid options include 'equatorial', 'polar', or, 'average'.")
            if self.polar_basis == 'polar':  # Polar
                self.polar_radius = POLAR_RADIUS_M
            elif self.polar_basis == 'average':
                self.polar_radius = AVERAGE_RADIUS_M
            elif self.polar_basis == 'equatorial':
                self.polar_radius = EQUATORIAL_RADIUS_M
            else:
                raise ValueError("Input polar_basis of '", self.polar_basis,
                                 "' is INVALID! Valid options include 'equatorial', 'polar', or, 'average'.")
        if self.grid_size <= 0:  # Check that the grid size is positive
            raise ValueError("Input grid_size of '", self.grid_size,
                             "' is INVALID! Value must be greater than 0.")

    def _getDeltas(self):
        """
        Computes the approximate longitudinal (delta_lon) and latitudinal (delta_lat) degree shifts required
        to achieve the "Orthodromic" distance set forth by class instance configurations (grid size, etc.).
        """
        # "Orthodromic" distance between any two points separated by a single degree change along both equatorial radius
        # (lon) and polar radius (lat) respectively.
        lon_unit_dist_per_deg = (self.equatorial_radius * math.pi) / 180
        lat_unit_dist_per_deg = (self.polar_radius * math.pi) / 180

        # Degree shift required to move the class instance self.grid_size distance amount instead of single km/mile
        self.delta_lon = self.grid_size / lon_unit_dist_per_deg
        self.delta_lat = self.grid_size / lat_unit_dist_per_deg

    def printConfig(self):
        """
        Public function for viewing class instance parameters that prints the class attribute values line by line in the
        terminal.
        """
        print("------------------------------------------------------")
        print("MPR instance configuration details")
        print("------------------------------------------------------")
        units = "km" if self.kilometers else "m"
        print("Grid size: ", self.grid_size, units)
        print("Radius: (", self.equatorial_basis, " = ", self.equatorial_radius, units, ", ",
              self.polar_basis, " = ", self.polar_radius, units, ")")
        print("Delta lat: ", self.delta_lat, " degrees")
        print("Delta lon: ", self.delta_lon, " degrees")
        print("------------------------------------------------------")

    def _createPlot(self, result: pd.DataFrame):
        """
        Private function for plotting mapPointReduce() results when mapPointReduce()'s parameter plot=True. Used for
        demonstration, testing, and verification. NOTE: can become very time-consuming when size of 'data' becomes
        relatively large and/or parameter configurations such as grid size, etc. become harder to plot
        (e.g. huge value ranges and small grid size, etc.)
        :param result: pandas Dataframe of mapPointReduce() results in form
            columns=["latitude", "longitude", "lat_mid", "lon_mid", "x_index", "y_index"]
        """
        # Custom plot params for easier viewing of plots
        plt.rcParams['figure.figsize'] = (10.0, 8.0)
        plt.rcParams['font.size'] = 16
        plt.rcParams['lines.linewidth'] = 2
        plt.rcParams['lines.linestyle'] = '-'

        # Grid size related plot preparations
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        # Capture ranges +/- lat and lon value ranges up to maximum for cleaner plot results
        lon_min = max(result["longitude"].min() - self.delta_lon, -180)
        lon_max = min(result["longitude"].max() + self.delta_lon, 180)
        lat_min = max(result["latitude"].min() - self.delta_lon, -90)
        lat_max = min(result["latitude"].max() + self.delta_lon, 90)

        # Major ticks (where values are labeled along axes) every delta * major_factor,
        # Minor ticks (where grid lines are but no labeling) every delta
        lon_factor = 1  # step size as to how frequently major ticks occur
        lon_minor_ticks_high = np.arange(0, 180, self.delta_lon)
        lon_minor_ticks_low = np.flip(lon_minor_ticks_high * -1)
        lon_minor_ticks = np.concatenate((lon_minor_ticks_low[:-1], lon_minor_ticks_high))
        lon_major_ticks = lon_minor_ticks[::lon_factor]

        ax.set_xticks(lon_major_ticks)
        ax.set_xticks(lon_minor_ticks, minor=True)

        lat_factor = 1  # step size as to how frequently major ticks occur
        lat_minor_ticks_high = np.arange(0, 90, self.delta_lat)
        lat_minor_ticks_low = np.flip(lat_minor_ticks_high * -1)
        lat_minor_ticks = np.concatenate((lat_minor_ticks_low[:-1], lat_minor_ticks_high))
        lat_major_ticks = lat_minor_ticks[::lat_factor]

        ax.set_yticks(lat_major_ticks)
        ax.set_yticks(lat_minor_ticks, minor=True)

        # set custom grid lines for minor and major tick intervals
        ax.grid(which='minor', color='#CCCCCC', linestyle='--', alpha=0.2)
        ax.grid(which='major', color='#000000', linestyle=':', alpha=0.8)

        # Polt labeling
        plt.title("Map Points Reduction")
        plt.xlabel("Longitude (delta = " + str(f'{self.delta_lon:.5f}') + ")")  # show 5 decimal places accuracy
        plt.ylabel("Latitude (delta = " + str(f'{self.delta_lat:.5f}') + ")")

        # capture range of grid cell indices for iterating over in scatter color setting process
        min_x_index = int(result["x_index"].min())
        max_x_index = int(result["x_index"].max())
        min_y_index = int(result["y_index"].min())
        max_y_index = int(result["y_index"].max())

        # Create iterative of shuffled color combinations for easier plot viewing where each cell has one color more
        # likely distinguishable from neighboring cells
        colors = cm.rainbow(np.linspace(0, 1, result.shape[0]))
        np.random.shuffle(colors)
        colors = iter(colors)

        # iterate over each cell x_index, y_index pair combination and create two scatters of same color: 1 for
        # initial lat, lon values (smaller size) and 1 for assigned map reduce value (larger size)
        for x_index in range(min_x_index, max_x_index + 1, 1):
            for y_index in range(min_y_index, max_y_index + 1, 1):
                # select all entries assigned to same point (via grid cell indices)
                cell_group = result[(result["x_index"] == x_index) & (result["y_index"] == y_index)]
                if not cell_group.empty:  # if at least one sample assigned to that point
                    color = next(colors)  # get next new color from shuffled list

                    # plot original lat, lon values smaller size
                    plt.scatter(cell_group["longitude"], cell_group["latitude"], color=color, s=10)

                    # plot new assigned lat, lon values larger size
                    plt.scatter(cell_group["lon_mid"], cell_group["lat_mid"], color=color, s=100)
        plt.show()

    def mapPointReduce(self, data: pd.DataFrame, plot: bool = True):
        """
        Public function that performs Map Points Reduction (MPR) process on passed pandas Dataframe 'data' based on
        class instance configuration parameters and returns pandas Dataframe of passed 'data' with two new columns
        "lat_mid" and "lon_mid" that correspond to computed MPR results.
        :param data: pandas Dataframe on which to perform the MPR process
        :param plot: boolean whether to create plot of results
        :return: pandas Dataframe 'data' with MPR results appended in new columns named "lat_mid" and "lon_mid"
        """
        data_copy = data.copy()  # create copy data so don't change original
        lat_array = data_copy["latitude"].to_numpy()  # create numpy arrays of lat and lon values separated
        lon_array = data_copy["longitude"].to_numpy()

        # get midpoint lat and lon assignment values
        midpoints = self._getMidpoints(lat_values=lat_array, lon_values=lon_array, plot=plot)
        midpoints = data_copy.join(midpoints)  # join copy of data to found midpoints
        if plot:
            self._createPlot(midpoints)  # send info to plot function (with cell indices gotten from _getMidpoints())
            midpoints.drop('x_index', inplace=True, axis=1)  # remove indices info before returning result
            midpoints.drop('y_index', inplace=True, axis=1)
        return midpoints

    def _getMidpoints(self, lat_values: np.array, lon_values: np.array, plot: bool):
        """
        Private helper function for public mapPointReduce() that computes the middle (lat_mid, lon_mid) coordinates
        for (lat_values, lon_values) given pairs based on grid cell indexing determined by delta's resulting from class
        instance configuration parameters.
        :param lat_values: numpy array of data's "latitude" values
        :param lon_values: numpy array of data's "longitude" values
        :param plot: bool whether plotting was requested (plot=True) in mapPointReduce().
        :return: If plot = false, numpy array of middle (lat_mid, lon_mid) values based on assigned grid cell indexing
        in form [[lat 0, lon 0], [lat 1, lon 1], ..., [lat n, lon n]] given n (lat, lon) input pairs in form
        columns=["lat_mid", "lon_mid"].
        Else, numpy array of middle (lat_mid, lon_mid) values as described above with grid cell indexing concatenated to
        end given form columns=["lat_mid", "lon_mid", "x_index", "y_index"].
        """
        # 'mids' starts as cell indices then overridden to reflect middle (lat, lon) value of assigned grid cell indices
        cell_indices = self._getCellIndices(lat_values=lat_values, lon_values=lon_values)
        midpoints = np.copy(cell_indices)
        # NOTE: x_index corresponds to lon, y_index corresponds to lat (inverted form of tabular lat then lon ordering)
        # np.where() functions as vectorized ternary operator, using for using different grid cell middle formulas, one
        # for negatively indexed and another for positively
        # find lat_mid (first column) based on y_indices (second column)
        midpoints[:, 0] = np.where(cell_indices[:, 1] < 0,  # condition
                                   (self.delta_lon * (cell_indices[:, 1] - 1)) - (self.delta_lon / 2),  # IF condition
                                   (self.delta_lon * (cell_indices[:, 1] + 1)) - (self.delta_lon / 2))  # ELSE
        # find lon_mid (second column) based on x_indices (first column)
        midpoints[:, 1] = np.where(cell_indices[:, 0] < 0,  # condition
                                   (self.delta_lat * (cell_indices[:, 0] - 1)) - (self.delta_lat / 2),  # IF condition
                                   (self.delta_lat * (cell_indices[:, 0] + 1)) - (self.delta_lat / 2))  # ELSE
        if plot:  # add cell indices to result so that they can be used during plot process to more easily color groups
            midpoints = pd.DataFrame(
                np.concatenate((midpoints, cell_indices), axis=1), columns=["lat_mid", "lon_mid", "x_index", "y_index"])
        else:
            midpoints = pd.DataFrame(midpoints, columns=["lat_mid", "lon_mid"])
        return midpoints

    def _getCellIndices(self, lat_values: np.array, lon_values: np.array):
        """
        Private helper function for _getMidpoints() that computes ant returns the grid cell indices (X, Y) for all
        latitude (lat_values) and longitude (lon_values) pairs based on implicit overlaying grid of self.delta_lon width
        and self.delta_lat height built out from (0, 0) position.
        :param lat_values: numpy array of latitude values
        :param lon_values: numpy array of longitude values
        :return: numpy array of cell indices [[x0, y0], [x1, y1], ..., [xn, yn]] given n (lat, lon) input pairs.
        """
        x_indices = np.floor(lon_values / self.delta_lon).reshape(-1, 1)
        y_indices = np.floor(lat_values / self.delta_lat).reshape(-1, 1)
        return np.concatenate((x_indices, y_indices), axis=1)
