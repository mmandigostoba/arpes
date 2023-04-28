"""Provides extremely fast 2D and 3D linear interpolation.

This is used for momentum conversion in place of the scipy
GridInterpolator where it is possible to do so. It is many many 
times faster than the grid interpolator and together with other optimizations
resulted in a 50x improvement in the momentum conversion time for
ARPES data in PyARPES.
"""
import numba
from dataclasses import dataclass
from typing import List, Union
import math
import numpy as np

__all__ = [
    "Interpolator",
]


@numba.njit
def to_fractional_coordinate(coord, initial, delta):
    return (coord - initial) / delta


@numba.njit
def _i1d(xd, c0, c1):
    return c0 * (1 - xd) + c1 * xd


@numba.njit
def raw_lin_interpolate_1d(xd, c0, c1):
    return _i1d(xd, c0, c1)


@numba.njit
def raw_lin_interpolate_2d(xd, yd, c00, c01, c10, c11):
    # xd, yd is x_delta, y_delta
    # project to 1D
    c0 = _i1d(xd, c00, c10)
    c1 = _i1d(xd, c01, c11)

    return _i1d(yd, c0, c1)


@numba.njit
def raw_lin_interpolate_3d(xd, yd, zd, c000, c001, c010, c100, c011, c101, c110, c111):
    # project to 2D
    c00 = _i1d(xd, c000, c100)
    c01 = _i1d(xd, c001, c101)
    c10 = _i1d(xd, c010, c110)
    c11 = _i1d(xd, c011, c111)

    # project to 1D
    c0 = _i1d(yd, c00, c10)
    c1 = _i1d(yd, c01, c11)

    return _i1d(zd, c0, c1)

@numba.njit
def raw_lin_interpolate_4d(xd, yd, zd, wd,
                           c0000, c0001, c0010, c0011,
                           c0100, c0101, c0110, c0111,
                           c1000, c1001, c1010, c1011,
                           c1100, c1101, c1110, c1111):
    # project to 3D
    c000 = _i1d(xd, c0000, c1000)
    c001 = _i1d(xd, c0001, c1001)
    c010 = _i1d(xd, c0010, c1010)
    c011 = _i1d(xd, c0011, c1011)
    c100 = _i1d(xd, c0100, c1100)
    c101 = _i1d(xd, c0101, c1101)
    c110 = _i1d(xd, c0110, c1110)
    c111 = _i1d(xd, c0111, c1111)
    
    # project to 2D
    c00 = _i1d(yd, c000, c100)
    c01 = _i1d(yd, c001, c101)
    c10 = _i1d(yd, c010, c110)
    c11 = _i1d(yd, c011, c111)

    # project to 1D
    c0 = _i1d(zd, c00, c10)
    c1 = _i1d(zd, c01, c11)

    return _i1d(wd, c0, c1)


@numba.njit
def lin_interpolate_4d(data, ix, iy, iz, iw,
                       ixp, iyp, izp, iwp,
                       xd, yd, zd, wd):
    return raw_lin_interpolate_4d(
        xd,
        yd,
        zd,
        wd,
        data[ix][iy][iz][iw],
        data[ix][iy][iz][iwp],
        data[ix][iy][izp][iw],
        data[ix][iy][izp][iwp],
        data[ix][iyp][iz][iw],
        data[ix][iyp][iz][iwp],
        data[ix][iyp][izp][iw],
        data[ix][iyp][izp][iwp],
        data[ixp][iy][iz][iw],
        data[ixp][iy][iz][iwp],
        data[ixp][iy][izp][iw],
        data[ixp][iy][izp][iwp],
        data[ixp][iyp][iz][iw],
        data[ixp][iyp][iz][iwp],
        data[ixp][iyp][izp][iw],
        data[ixp][iyp][izp][iwp],
    )

@numba.njit
def lin_interpolate_3d(data, ix, iy, iz, ixp, iyp, izp, xd, yd, zd):
    return raw_lin_interpolate_3d(
        xd,
        yd,
        zd,
        data[ix][iy][iz],
        data[ix][iy][izp],
        data[ix][iyp][iz],
        data[ixp][iy][iz],
        data[ix][iyp][izp],
        data[ixp][iy][izp],
        data[ixp][iyp][iz],
        data[ixp][iyp][izp],
    )


@numba.njit
def lin_interpolate_2d(data, ix, iy, ixp, iyp, xd, yd):
    return raw_lin_interpolate_2d(
        xd,
        yd,
        data[ix][iy],
        data[ix][iyp],
        data[ixp][iy],
        data[ixp][iyp],
    )


@numba.njit(parallel=True)
def interpolate_4d(
    data,
    output,
    lower_corner_x,
    lower_corner_y,
    lower_corner_z,
    lower_corner_w,
    delta_x,
    delta_y,
    delta_z,
    delta_w,
    shape_x,
    shape_y,
    shape_z,
    shape_w,
    x,
    y,
    z,
    w,
    fill_value=np.nan,
):
    for i in numba.prange(len(x)):
        if np.isnan(x[i]) or np.isnan(y[i]) or np.isnan(z[i]) or np.isnan(w[i]):
            output[i] = fill_value
            continue

        ix = to_fractional_coordinate(x[i], lower_corner_x, delta_x)
        iy = to_fractional_coordinate(y[i], lower_corner_y, delta_y)
        iz = to_fractional_coordinate(z[i], lower_corner_z, delta_z)
        iw = to_fractional_coordinate(w[i], lower_corner_w, delta_w)

        if (ix < 0 or iy < 0 or iz < 0 or iw < 0
            or ix >= shape_x or iy >= shape_y or iz >= shape_z or iw >= shape_w):
            output[i] = fill_value
            continue

        iix, iiy, iiz, iiw = math.floor(ix), math.floor(iy), math.floor(iz), math.floor(iw)
        iixp, iiyp, iizp, iiwp = (
            min(iix + 1, shape_x - 1),
            min(iiy + 1, shape_y - 1),
            min(iiz + 1, shape_z - 1),
            min(iiw + 1, shape_w - 1),
        )
        xd, yd, zd, wd = ix - iix, iy - iiy, iz - iiz, iw - iiw

        output[i] = lin_interpolate_4d(data, iix, iiy, iiz, iiw,
                                       iixp, iiyp, iizp, iiwp,
                                       xd, yd, zd, wd)


@numba.njit(parallel=True)
def interpolate_3d(
    data,
    output,
    lower_corner_x,
    lower_corner_y,
    lower_corner_z,
    delta_x,
    delta_y,
    delta_z,
    shape_x,
    shape_y,
    shape_z,
    x,
    y,
    z,
    fill_value=np.nan,
):
    for i in numba.prange(len(x)):
        if np.isnan(x[i]) or np.isnan(y[i]) or np.isnan(z[i]):
            output[i] = fill_value
            continue

        ix = to_fractional_coordinate(x[i], lower_corner_x, delta_x)
        iy = to_fractional_coordinate(y[i], lower_corner_y, delta_y)
        iz = to_fractional_coordinate(z[i], lower_corner_z, delta_z)

        if ix < 0 or iy < 0 or iz < 0 or ix >= shape_x or iy >= shape_y or iz >= shape_z:
            output[i] = fill_value
            continue

        iix, iiy, iiz = math.floor(ix), math.floor(iy), math.floor(iz)
        iixp, iiyp, iizp = (
            min(iix + 1, shape_x - 1),
            min(iiy + 1, shape_y - 1),
            min(iiz + 1, shape_z - 1),
        )
        xd, yd, zd = ix - iix, iy - iiy, iz - iiz

        output[i] = lin_interpolate_3d(data, iix, iiy, iiz, iixp, iiyp, iizp, xd, yd, zd)


@numba.njit(parallel=True)
def interpolate_2d(
    data,
    output,
    lower_corner_x,
    lower_corner_y,
    delta_x,
    delta_y,
    shape_x,
    shape_y,
    x,
    y,
    fill_value=np.nan,
):
    for i in numba.prange(len(x)):
        if np.isnan(x[i]) or np.isnan(y[i]):
            output[i] = fill_value
            continue

        ix = to_fractional_coordinate(x[i], lower_corner_x, delta_x)
        iy = to_fractional_coordinate(y[i], lower_corner_y, delta_y)

        if ix < 0 or iy < 0 or ix >= shape_x - 1 or iy >= shape_y - 1:
            output[i] = fill_value
            continue

        iix, iiy = math.floor(ix), math.floor(iy)
        iixp, iiyp = (
            min(iix + 1, shape_x - 1),
            min(iiy + 1, shape_y - 1),
        )
        xd, yd = ix - iix, iy - iiy

        output[i] = lin_interpolate_2d(data, iix, iiy, iixp, iiyp, xd, yd)


@dataclass
class Interpolator:
    """Provides a Pythonic interface to fast gridded linear interpolation.

    More or less a drop-in replacement for scipy's RegularGridInterpolator,
    but much faster at the expense of not supporting any extrapolation.
    """

    lower_corner: List[float]
    delta: List[float]
    shape: List[int]
    data: np.ndarray

    def __post_init__(self):
        """Convert data to floating point representation.

        Because we do linear not nearest neighbor interpolation this should be safe
        always.
        """
        self.data = self.data.astype(np.float64, copy=False)

    @classmethod
    def from_arrays(cls, xyz: List[np.ndarray], data: np.ndarray):
        """Initializes the interpreter from a coordinate and data array.

        Args:
            xyz: A list of the coordinate arrays. Should be length 2 or 3
              because we provide 2D and 3D coordinate interpolation.
            data: The value of the interpolated function at the coordinate in `xyz`
        """
        lower_corner = [xi[0] for xi in xyz]
        delta = [xi[1] - xi[0] for xi in xyz]
        shape = [len(xi) for xi in xyz]
        return cls(lower_corner, delta, shape, data)

    def __call__(self, xi: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """Performs linear interpolation at the coordinates given by `xi`.

        Whether 2D or 3D interpolation is used depends on the dimensionality of `xi` and
        `self.data` but of course they must match one another.

        Args:
            xi: A list or stacked array of the coordinates. Provides a [d, k] array
              of k points each with d dimensions/indices.

        Returns:
            The interpolated values f(x_i) at each point x_i, as a length k scalar array.
        """
        if isinstance(xi, np.ndarray):
            xi = xi.astype(np.float64, copy=False)
            xi = [xi[:, i] for i in range(self.data.ndim)]
        else:
            xi = [xii.astype(np.float64, copy=False) for xii in xi]

        output = np.zeros_like(xi[0])

        interpolator = {
            4: interpolate_4d,
            3: interpolate_3d,
            2: interpolate_2d,
        }[self.data.ndim]

        interpolator(
            self.data,
            output,
            *self.lower_corner,
            *self.delta,
            *self.shape,
            *xi,
        )

        return output
