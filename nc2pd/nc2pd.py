"""
nc2pd
~~~~~

A thin python-netCDF4 wrapper to turn netCDF files into pandas data
structures, with a focus on extracting time series from regularly
spatial gridded data (with the ability to interpolate spatially).

Copyright 2015 Stefan Pfenninger

License: MIT (see LICENSE file)

"""

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import itertools

import numpy as np
import pandas as pd
from scipy import interpolate, ndimage
from netCDF4 import Dataset, num2date


class NetCDFDataset(object):
    """NetCDFDataset"""
    def __init__(self, path):
        super(NetCDFDataset, self).__init__()
        self.path = path
        self.rootgrp = Dataset(self.path)
        # Determine latitute and longitude variable names
        self.latlon_names = self._latlon_names()
        # Generate datetime labels for the time variable
        # Also sets self.time_name
        self.datetimes = self._datetime_labels()
        # Get array of latitude and longitude values
        lat_name, lon_name = self.latlon_names
        self.lon_array = self.rootgrp.variables[lon_name][:]
        self.lat_array = self.rootgrp.variables[lat_name][:]
        # Additional dimension slices set up internally as needed
        self.dim_slices_collapse = {}
        self.dim_slices_select = {}

    def _latlon_names(self):
        """Determines the lat/lon variable names in the dataset"""
        if 'latitude' in self.rootgrp.variables:
            lat_name = 'latitude'
            lon_name = 'longitude'
        elif 'lat' in self.rootgrp.variables:
            lat_name = 'lat'
            lon_name = 'lon'
        elif 'XDim' in self.rootgrp.variables:
            lat_name = 'YDim'
            lon_name = 'XDim'
        else:
            raise ValueError('Cannot determine lat and lon variable names')
        return (lat_name, lon_name)

    def _datetime_labels(self):
        """Return datetime labels for the dataset"""
        if ('dfb' in self.rootgrp.variables
                and 'hour' in self.rootgrp.variables):
            # solargis data has dimensions ('dfb', 'hour', 'latitude', 'longitude')
            # we must do some manual processing of the time dimension
            # dfb - 1 to account for 00:00 representation of 24:00
            # pushing us into the next day
            days = num2date(self.rootgrp.variables['dfb'][:] - 1,
                            'days since 1980-01-01')
            dt_from = '{} 00:00'.format(days[0].strftime('%Y-%m-%d'))
            dt_to = '{} 23:00'.format(days[-1].strftime('%Y-%m-%d'))
            dates = pd.date_range(dt_from, dt_to, freq='H')
            # Set an additional slice on hour internally
            self.dim_slices_collapse['hour'] = slice(None, None, None)  # [::]
            self.time_name = 'dfb'
        else:
            try:
                time_name = 'time'
                timevar = self.rootgrp.variables[time_name]
            except AttributeError:
                try:
                    time_name = 'TIME'
                    timevar = self.rootgrp.variables[time_name]
                except AttributeError:
                    raise ValueError('Cannot find time variable.')
            self.time_name = time_name
            try:
                timevar_units = timevar.units.decode()
            except AttributeError:
                timevar_units = timevar.units
            dates = num2date(timevar[:], timevar_units, calendar='standard')
        labels = pd.Series(range(len(dates)), index=dates)
        return labels

    def _find_coordinates(self, lat, lon, bounds=False):
        """
        Finds the index of given lat/lon pair in the dataset.

        Uses binary search to find closest coordinates if the exact ones
        don't exist.

        Parameters
        ----------
        lat : float
            latitude
        lon : float
            longitude

        Returns
        -------
        x, y : 4-tuple
            x and y (lon and lat) coordinate indices

        """
        def _find_closest(array, value, bounds):
            """Searches array for value and returns the index of the entry
            closest to value."""
            if bounds:
                pos = np.searchsorted(array, value)
                return (pos - 1, pos)
            else:
                return (np.abs(array - value)).argmin()

        if lon in self.lon_array:
            x = np.argmax(self.lon_array == lon)
            if bounds:
                x = (x, x)
        else:
            x = _find_closest(self.lon_array, lon, bounds)
        if lat in self.lat_array:
            y = np.argmax(self.lat_array == lat)
            if bounds:
                y = (y, y)
        else:
            y = _find_closest(self.lat_array, lat, bounds)
        # Return either a single x, y pair or a list of pairs: [(x, y)]
        if bounds:
            return list(zip(x, y))
        else:
            return (x, y)

    def get_gridpoints(self, latlon_pairs, bounds=False):
        """Take a list of lat-lon pairs and return a list of x-y indices."""
        points = [self._find_coordinates(lat, lon, bounds)
                  for lat, lon in latlon_pairs]
        return [i for i in itertools.chain.from_iterable(points)]

    def get_timerange(self, start=None, end=None, include_end=True):
        """
        Take a start and end datetime and return a time index range.

        If include_end is True, the returned range is 1 longer so that
        the final timestep given in the range is included in slicing.

        If the desired end point is not found in the data, the most recent
        available end point is used.

        """
        if start:
            try:
                start_idx = self.datetimes[start].ix[0]
            except AttributeError:  # because it's a single value already
                start_idx = self.datetimes[start]
        else:
            start_idx = self.datetimes.ix[0]
        if end:
            try:
                end_idx = self.datetimes[end].ix[-1]
            except AttributeError:  # because it's a single value already
                end_idx = self.datetimes[end]
            except IndexError:  # because we've hit a missing datetime entry
                # First get closest available end index
                end_idx = np.argmin(np.abs(self.datetimes.index.to_pydatetime() -
                                           pd.datetools.parse(end)))
                # Now check if this is beyond the desired end date, and if so,
                # move back one in the list of existing datetimes, which
                # will put us within the desired endpoint (given that the
                # desired endpoint didn't exist in the first place!)
                if self.datetimes.index[end_idx] > pd.datetools.parse(end):
                    end_idx = end_idx - 1
        else:
            end_idx = self.datetimes.ix[-1]
        if include_end:
            end_idx += 1
        return (start_idx, end_idx)

    def read_data(self, variable, x_range=None, y_range=None,
                  time_range=None, fixed_dims={},
                  friendly_labels=False):
        """
        Return a panel of data with the dimensions [time, lat, lon], i.e.
        items are time, major_axis is latitude, minor_axis is longitude.

        Parameters
        ----------
        variable : str
            name of variable
        x_range : int or (int, int), default None
            range of x grid points to select, if None, entire x range is used
        y_range : int or (int, int), default None
            range of y grid points to select, if None, entire y range is used
        time_range : int or (int, int), default None
            range of timesteps to select, if None, entire time range is used
        fixed_dims : dict, default {}
            map selections to other dimensions that may exist in the data,
            e.g. {'level': 0}
        friendly_labels : bool, default False
            if True, sets the axis labels to datetimes, latitudes and
            longitudes, instead of just integer indices

        """
        # Helpers
        slice_all = slice(None, None, None)

        def add_slice(setting, var_slice):
            if not setting:
                slicer = slice_all
            else:
                if isinstance(setting, int) or isinstance(setting, np.integer):
                    slicer = slice(setting, setting + 1)
                else:  # Assume two or more integers
                    slicer = slice(*setting)
            var_slice.append(slicer)
            return slicer

        # Start work
        var = self.rootgrp.variables[variable]
        var_slice = []
        dim_pos = 0
        # Transposition so that the panel order is always time, lat, lon
        transposition = [None, None, None]
        for dim in var.dimensions:
            # 1. check if it's time, lat or lon name
            #    and assign appropriate slice if so
            if dim == self.time_name:
                time_slice = add_slice(time_range, var_slice)
                transposition[0] = dim_pos
                dim_pos += 1
            elif dim == self.latlon_names[0]:  # lat --> y
                y_slice = add_slice(y_range, var_slice)
                transposition[1] = dim_pos
                dim_pos += 1
            elif dim == self.latlon_names[1]:  # lon --> x
                x_slice = add_slice(x_range, var_slice)
                transposition[2] = dim_pos
                dim_pos += 1
            # 2. check if it's in self.dim_slices
            elif dim in self.dim_slices_collapse:
                var_slice.append(self.dim_slices_collapse[dim])
                # FIXME after taking var[var_slice], will also need
                # to collapse all dim_slices_collapse simensions,
                # or else reading e.g. solargis files won't work
                raise NotImplementedError('well, that did not work!')
            elif dim in self.dim_slices_select:
                var_slice.append(self.dim_slices_select[dim])
            # 3. check if it's in fixed_dims
            elif dim in fixed_dims:
                var_slice.append(fixed_dims[dim])
            # 4. else, raise a KeyError or something
            else:
                raise KeyError('Dimension `{}` unknown'.format(dim))
        panel = pd.Panel(var[var_slice]).transpose(*transposition)
        if friendly_labels:
            panel.items = self.datetimes.index[time_slice]
            panel.major_axis = self.lat_array[y_slice]
            panel.minor_axis = self.lon_array[x_slice]
        return panel

    def read_timeseries(self, variable, latlon_pairs,
                        start=None, end=None,
                        buffer_size=0,
                        fixed_dims={},
                        return_metadata=False):
        """
        Return a time series for each given lat-lon pair.

        Parameters
        ----------
        variable : str
            name of variable
        latlon_pairs : list of (lat, lon) tuples
            list of (lat, lon) tuples
        start : str, default None
            datetime string of the form 'YYYY-MM-DD hh:mm' or similar
        end : str, default None
            datetime string, like for start
        fixed_dims : dict, default {}
            map selections to other dimensions that may exist in the data,
            e.g. {'level': 0}

        Returns
        -------
        data : one or two pandas DataFrames
            the first DataFrame contains each requested lat-lon pair
            as a column
            if return_metadata is True, the second DataFrame maps from
            the requested latitudes/longitudes to grid points and
            their latitudes/longitudes

        """
        gridpoints = self.get_gridpoints(latlon_pairs)
        timerange = self.get_timerange(start, end)

        # Data
        dfs = []
        for x, y in gridpoints:
            if buffer_size:
                x_slice = (x - buffer_size, x + 1 + buffer_size)
                y_slice = (y - buffer_size, y + 1 + buffer_size)
            else:
                x_slice = x
                y_slice = y
            panel = self.read_data(variable,
                                   x_range=x_slice,
                                   y_range=y_slice,
                                   time_range=timerange,
                                   fixed_dims=fixed_dims,
                                   friendly_labels=True)
            dfs.append(panel.to_frame().T)

        # Metadata
        md = pd.DataFrame(latlon_pairs, columns=['lat', 'lon'])
        grid_cols = list(zip(*gridpoints))
        md['y_gridpoint'] = grid_cols[1]
        md['x_gridpoint'] = grid_cols[0]
        md['lat_gridpoint'] = [self.lat_array[i] for i in md['y_gridpoint']]
        md['lon_gridpoint'] = [self.lon_array[i] for i in md['x_gridpoint']]

        data = pd.concat(dfs, axis=1)

        if return_metadata:
            return (data, md)
        else:
            return data

    def read_boundingbox(self, variable, latlon_pairs,
                         start=None, end=None,
                         buffer_size=0,
                         fixed_dims={}):
        """
        Return a time-lat-lon panel encompassing all the given lat-lon pairs,
        with a surrounding buffer.

        Parameters
        ----------
        variable : str
            name of variable
        latlon_pairs : list of (lat, lon) tuples
            list of (lat, lon) tuples
        start : str, default None
            datetime string of the form 'YYYY-MM-DD hh:mm' or similar
        end : str, default None
            datetime string, like for start
        buffer_size : int, default 0
            Grid points by which to extend the bounding box around the
            outermost points. Set to 0 to disable.
        fixed_dims : dict, default {}
            map selections to other dimensions that may exist in the data,
            e.g. {'level': 0}

        """
        gridpoints = self.get_gridpoints(latlon_pairs, bounds=True)
        timerange = self.get_timerange(start, end)

        # Get bounding box
        x, y = list(zip(*gridpoints))
        x_slice = (min(x) - buffer_size, max(x) + 1 + buffer_size)
        y_slice = (min(y) - buffer_size, max(y) + 1 + buffer_size)

        panel = self.read_data(variable,
                               x_range=x_slice,
                               y_range=y_slice,
                               time_range=timerange,
                               fixed_dims=fixed_dims,
                               friendly_labels=True)
        return panel

    def read_interpolated_timeseries(self, variable, latlon_pairs,
                                     start=None, end=None,
                                     buffer_size=0,
                                     order=1, **kwargs):
        """
        Return an interpolated time series for each given lat-lon pair.

        Parameters
        ----------
        variable : str
            name of variable
        latlon_pairs : list of (lat, lon) tuples
            list of (lat, lon) tuples
        start : str, default None
            datetime string of the form 'YYYY-MM-DD hh:mm' or similar
        end : str, default None
            datetime string, like for start
        buffer_size : int, default 1
            Grid points by which to extend the bounding box around the
            outermost points. Set to 0 to disable.
        order : int, default 1
            order of spline to use, 1 is linear
        **kwargs
            additional keyword args are passed to ndimage.map_coordinates

        Returns
        -------
        data : pandas DataFrame
            each requested lat-lon pair as a column

        """
        data = self.read_boundingbox(variable, latlon_pairs,
                                     start=start, end=end,
                                     buffer_size=buffer_size)

        return spatial_interpolation(data, latlon_pairs, order=1, **kwargs)


def spatial_interpolation(data, latlon_pairs, order=1, **kwargs):
    """
    Parameters
    ----------
    data : pandas Panel
        with dimensions time (items), lat (major_axis), lon(minor_axis)
    latlon : (float, float) tuple
        latitude and longitude for which to interpolate
    order : int, default 1
        order of spline to use, 1 is linear
    **kwargs
        additional keyword args are passed to ndimage.map_coordinates

    """
    # lat, lon to array dimensions y, z
    m = {}
    for var, dim in [('y', data.major_axis), ('z', data.minor_axis)]:
        try:
            m[var] = interpolate.interp1d(dim, list(range(len(dim))))
        except ValueError:  # Raised if there is only one entry
            m[var] = lambda x: 0  # 0 is the only index that exists

    # x dimension is time, we want ALL timesteps from the data
    x = list(range(len(data.items)))

    results = []

    # do the actual interpolation to y, z array coordinates
    for lat, lon in latlon_pairs:
        y = np.ones_like(x) * m['y'](lat)
        z = np.ones_like(x) * m['z'](lon)

        interp = ndimage.map_coordinates(data.as_matrix(), [x, y, z],
                                         order=order, **kwargs)

        results.append(pd.Series(interp))

    df = pd.concat(results, axis=1)
    df.index = data.items

    # If latlon_pairs contains only one pair, we ensure it's a tuple
    # because we'd try to set two columns otherwise
    if len(latlon_pairs) == 1:
        latlon_pairs = tuple(latlon_pairs)
    df.columns = latlon_pairs

    return df
