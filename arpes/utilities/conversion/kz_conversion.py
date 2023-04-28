"""Coordinate conversion classes for photon energy scans."""
import numpy as np
import numba
import xarray as xr

import arpes.constants
from typing import Any, Callable, Dict

from .base import CoordinateConverter, K_SPACE_BORDER, MOMENTUM_BREAKPOINTS
from .bounds_calculations import calculate_kp_kz_bounds, calculate_kx_ky_kz_bounds
from .kx_ky_conversion import _exact_arcsin, _small_angle_arcsin, _rotate_kx_ky, _safe_compute_k_tot

__all__ = ["ConvertKpKzV0", "ConvertKxKyKz", "ConvertKpKz"]


@numba.njit(parallel=True, cache=True)
def _kspace_to_hv(kp, kz, hv, energy_shift, is_constant_shift):
    """Efficiently perform the inverse coordinate transform to photon energy."""
    shift_ratio = 0 if is_constant_shift else 1

    for i in numba.prange(len(kp)):
        hv[i] = (
            arpes.constants.HV_CONVERSION * (kp[i] ** 2 + kz[i] ** 2)
            + energy_shift[i * shift_ratio]
        )

@numba.njit(parallel=True, cache=True)
def _kspace_to_hv_3d(kx, ky, kz, hv, energy_shift, is_constant_shift):
    """Efficiently perform the inverse coordinate transform to photon energy."""
    shift_ratio = 0 if is_constant_shift else 1

    for i in numba.prange(len(kx)):
        hv[i] = (
            arpes.constants.HV_CONVERSION * (kx[i] ** 2 + ky[i] ** 2 + kz[i] ** 2)
            + energy_shift[i * shift_ratio]
        )

@numba.njit(parallel=True, cache=True)
def _kp_to_polar(kinetic_energy, kp, phi, inner_potential, angle_offset):
    """Efficiently performs the inverse coordinate transform phi(hv, kp)."""
    for i in numba.prange(len(kp)):
        phi[i] = (
            np.arcsin(
                kp[i]
                / (arpes.constants.K_INV_ANGSTROM * np.sqrt(kinetic_energy[i] + inner_potential))
            )
            + angle_offset
        )


class ConvertKpKzV0(CoordinateConverter):
    """Implements inner potential broadcasted hv Fermi surfaces."""

    # TODO implement
    def __init__(self, *args, **kwargs):
        """TODO, implement this."""
        super(ConvertKpKzV0, self).__init__(*args, **kwargs)
        raise NotImplementedError


class ConvertKxKyKz(CoordinateConverter):
    """Implements 4D data volume conversion."""

    # To-Do: see if we need all the extra stuff in __init__ from the kxky converter
    def __init__(self, *args, **kwargs):
        """TODO, implement this."""
        super(ConvertKxKyKz, self).__init__(*args, **kwargs)
        self.hv = None
        self.phi = None
        
        self.k_tot = None
        # the angle perpendicular to phi as appropriate to the scan, this can be any of
        # psi, theta, beta
        self.perp_angle = None

        self.rkx = None
        self.rky = None

        # accept either vertical or horizontal, fail otherwise
        if not any(
            np.abs(arr.alpha - 
                   alpha_option) < (np.pi / 180) for alpha_option in [0, np.pi / 2]
        ):
            raise ValueError(
                "Must convert either vertical or horizontal slit data with this converter."
            )

        self.direct_angles = ("phi", 
                              [d for d in ["psi", "beta", "theta"] if d in arr.indexes][0])

        if self.direct_angles[1] != "psi":
            # psi allows for either orientation
            assert (self.direct_angles[1] in {"theta"}) != (not self.is_slit_vertical)

        # determine which other angles constitute equivalent sets
        opposite_direct_angle = "theta" if "psi" in self.direct_angles else "psi"
        if self.is_slit_vertical:

            self.parallel_angles = (
                "beta",
                opposite_direct_angle,
            )
        else:
            self.parallel_angles = (
                "theta",
                opposite_direct_angle,
            )
        
    def get_coordinates(
        self, resolution: dict = None, bounds: dict = None
    ) -> Dict[str, np.ndarray]:
        """Calculates appropriate coordinate bounds."""
        if resolution is None:
            resolution = {}
        if bounds is None:
            bounds = {}

        coordinates = super(ConvertKxKyKz, self).get_coordinates(resolution=resolution, 
                                                               bounds=bounds)

        ((kx_low, kx_high), 
         (ky_low, ky_high), 
         (kz_low, kz_high)) = calculate_kx_ky_kz_bounds(self.arr)
        
        if "kx" in bounds:
            kx_low, kx_high = bounds["kx"]
            
        if "ky" in bounds:
            ky_low, ky_high = bounds["ky"]

        if "kz" in bounds:
            kz_low, kz_high = bounds["kz"]
            
        kx_angle, ky_angle = self.direct_angles
        
        if self.is_slit_vertical:
            # phi actually measures along ky
            ky_angle, kx_angle = kx_angle, ky_angle

        len_ky_angle = len(self.arr.coords[ky_angle])
        len_kx_angle = len(self.arr.coords[kx_angle])

        inferred_kx_res = (kx_high - kx_low + 2 * K_SPACE_BORDER
                          ) / len(self.arr.coords[kx_angle])
        inferred_ky_res = (ky_high - ky_low + 2 * K_SPACE_BORDER
                          ) / len(self.arr.coords[ky_angle])

        '''inferred_kp_res = [b for b in MOMENTUM_BREAKPOINTS if b < inferred_kp_res][-1]'''

        # go a bit finer here because it would otherwise be very coarse
        inferred_kz_res = (kz_high - kz_low + 2 * K_SPACE_BORDER) / len(self.arr.coords["hv"])
        inferred_kz_res = [b for b in MOMENTUM_BREAKPOINTS if b < inferred_kz_res][-1]

        # upsample a bit if there aren't that many points along a certain axis
        try:
            inferred_kx_res = [b for b in MOMENTUM_BREAKPOINTS if b < inferred_kx_res][
                -2 if (len_kx_angle < 80) else -1
            ]
        except IndexError:
            inferred_kx_res = MOMENTUM_BREAKPOINTS[-2]
        try:
            inferred_ky_res = [b for b in MOMENTUM_BREAKPOINTS if b < inferred_ky_res][
                -2 if (len_ky_angle < 80) else -1
            ]
        except IndexError:
            inferred_ky_res = MOMENTUM_BREAKPOINTS[-2]
            
        coordinates["kx"] = np.arange(
            kx_low - K_SPACE_BORDER, kx_high + K_SPACE_BORDER, 
            resolution.get("kx", inferred_kx_res)
        )
        coordinates["ky"] = np.arange(
            ky_low - K_SPACE_BORDER, ky_high + K_SPACE_BORDER, 
            resolution.get("ky", inferred_ky_res)
        )
        
        '''coordinates["kp"] = np.arange(
            kp_low - K_SPACE_BORDER, kp_high + K_SPACE_BORDER, 
            resolution.get("kp", inferred_kp_res)
        )'''
        
        coordinates["kz"] = np.arange(
            kz_low - K_SPACE_BORDER, kz_high + K_SPACE_BORDER, 
            resolution.get("kz", inferred_kz_res)
        )

        '''base_coords = {k: v for k, 
                       v in self.arr.coords.items() if k not in ["eV", "phi", hv"]}'''
        
        base_coords = {
            k: v for k, 
            v in self.arr.coords.items()
            if k not in ["eV", "phi", "psi", "theta", "beta", "alpha", "chi", "hv"]
        }

        coordinates.update(base_coords)

        return coordinates

    def compute_k_tot(self, binding_energy: np.ndarray) -> None:
        """Compute the total momentum (inclusive of kz) at different binding energies."""
        self.k_tot = _safe_compute_k_tot(self.hv.data.obj, 
                                         self.arr.S.work_function, 
                                         binding_energy)

    def conversion_for(self, dim: str) -> Callable:
        """Looks up the appropriate momentum-to-angle conversion routine by dimension name."""

        def with_identity(*args, **kwargs):
            return self.identity_transform(dim, *args, **kwargs)

        return {
            "eV": self.kspace_to_BE,
            "phi": self.kspace_to_phi,
            "theta": self.kspace_to_perp_angle,
            "psi": self.kspace_to_perp_angle,
            "beta": self.kspace_to_perp_angle,
            "hv": self.kspace_to_hv,
        }.get(dim, with_identity)

    @property
    def needs_rotation(self) -> bool:
        """Whether we need to rotate the momentum coordinates when converting to angle."""
        # force rotation when greater than 0.5 deg
        return np.abs(self.arr.S.lookup_offset_coord("chi")) > (0.5 * np.pi / 180)

    def rkx_rky(self, kx, ky):
        """Returns the rotated kx and ky values when we are rotating by nonzero chi."""
        if self.rkx is not None:
            return self.rkx, self.rky

        chi = self.arr.S.lookup_offset_coord("chi")

        self.rkx = np.zeros_like(kx)
        self.rky = np.zeros_like(ky)
        _rotate_kx_ky(kx, ky, self.rkx, self.rky, chi)

        return self.rkx, self.rky

    def kspace_to_hv(
        self, 
        binding_energy: np.ndarray, 
        kx: np.ndarray,
        ky: np.ndarray,
        kz: np.ndarray, 
        *args: Any, 
        **kwargs: Any
    ) -> np.ndarray:
        """Converts from momentum back to the raw photon energy."""
        if self.hv is None:
            inner_v = self.arr.S.inner_potential
            wf = self.arr.S.work_function

            is_constant_shift = True
            if not isinstance(binding_energy, np.ndarray):
                is_constant_shift = True
                binding_energy = np.array([binding_energy])

            self.hv = np.zeros_like(kx)
            _kspace_to_hv(kx, ky, kz, self.hv, -inner_v - binding_energy + wf, is_constant_shift)
        
        print('hv')
        print(np.min(self.hv))
        print(np.max(self.hv))

        return self.hv
    
    def kspace_to_phi(
        self, 
        binding_energy: np.ndarray, 
        kx: np.ndarray, 
        ky: np.ndarray, 
        kz: np.ndarray,
        *args: Any, 
        **kwargs: Any
    ) -> np.ndarray:
        """Converts from momentum back to the analyzer angular axis."""
        if self.phi is not None:
            return self.phi

        if self.k_tot is None:
            self.compute_k_tot(binding_energy)

        if self.needs_rotation:
            kx, ky = self.rkx_rky(kx, ky)

        # This can be condensed but it is actually better not to condense it:
        # In this format, we can easily compare to the raw coordinate conversion functions
        # from Mathematica in order to adjust signs, etc.
        scan_angle = self.direct_angles[1]
        self.phi = np.zeros_like(self.k_tot)
        offset = self.arr.S.phi_offset + self.arr.S.lookup_offset_coord(
            self.parallel_angles[0])

        par_tot = isinstance(self.k_tot, np.ndarray) and len(self.k_tot) != 1
        assert len(self.k_tot) == len(self.phi) or len(self.k_tot) == 1

        if scan_angle == "psi":
            if self.is_slit_vertical:
                _exact_arcsin(ky, kx, self.k_tot, self.phi, offset, par_tot, False)
            else:
                _exact_arcsin(kx, ky, self.k_tot, self.phi, offset, par_tot, False)
        elif scan_angle == "beta":
            # vertical slit
            _small_angle_arcsin(kx, self.k_tot, self.phi, offset, par_tot, False)
            print('phi')
            print(np.min(self.phi))
            print(np.max(self.phi))
        elif scan_angle == "theta":
            # vertical slit
            _small_angle_arcsin(ky, self.k_tot, self.phi, offset, par_tot, False)
        else:
            raise ValueError(
                "No recognized scan angle found for {}".format(self.parallel_angles[1])
            )

        try:
            self.phi = self.calibration.correct_detector_angle(eV=binding_energy, 
                                                               phi=self.phi)
        except:
            pass

        return self.phi

    def kspace_to_perp_angle(
        self, 
        binding_energy: np.ndarray, 
        kx: np.ndarray, 
        ky: np.ndarray,
        kz: np.ndarray,
        *args: Any, 
        **kwargs: Any
    ) -> np.ndarray:
        """Converts from momentum back to the scan angle perpendicular to the analyzer."""
        if self.perp_angle is not None:
            return self.perp_angle

        if self.k_tot is None:
            self.compute_k_tot(binding_energy)

        if self.needs_rotation:
            kx, ky = self.rkx_rky(kx, ky)

        scan_angle = self.direct_angles[1]
        self.perp_angle = np.zeros_like(self.k_tot)

        par_tot = isinstance(self.k_tot, np.ndarray) and len(self.k_tot) != 1
        assert len(self.k_tot) == len(self.perp_angle) or len(self.k_tot) == 1

        if scan_angle == "psi":
            if self.is_slit_vertical:
                offset = self.arr.S.psi_offset - self.arr.S.lookup_offset_coord(
                    self.parallel_angles[1]
                )
                _small_angle_arcsin(kx, self.k_tot, self.perp_angle, offset, par_tot, True)
            else:
                offset = self.arr.S.psi_offset + self.arr.S.lookup_offset_coord(
                    self.parallel_angles[1]
                )
                _small_angle_arcsin(ky, self.k_tot, self.perp_angle, offset, par_tot, False)
        elif scan_angle == "beta":
            offset = self.arr.S.beta_offset + self.arr.S.lookup_offset_coord(
                self.parallel_angles[1]
            )
            _exact_arcsin(ky, kx, self.k_tot, self.perp_angle, offset, par_tot, True)
            print('beta')
            print(np.min(self.perp_angle))
            print(np.max(self.perp_angle))
        elif scan_angle == "theta":
            offset = self.arr.S.theta_offset - self.arr.S.lookup_offset_coord(
                self.parallel_angles[1]
            )
            _exact_arcsin(kx, ky, self.k_tot, self.perp_angle, offset, par_tot, True)
        else:
            raise ValueError(
                "No recognized scan angle found for {}".format(self.parallel_angles[1])
            )

        return self.perp_angle
    



class ConvertKpKz(CoordinateConverter):
    """Implements single angle photon energy scans."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Cache the photon energy coordinate we calculate backwards from kz."""
        super(ConvertKpKz, self).__init__(*args, **kwargs)
        self.hv = None
        self.phi = None

    def get_coordinates(
        self, resolution: dict = None, bounds: dict = None
    ) -> Dict[str, np.ndarray]:
        """Calculates appropriate coordinate bounds."""
        if resolution is None:
            resolution = {}
        if bounds is None:
            bounds = {}

        coordinates = super(ConvertKpKz, self).get_coordinates(resolution=resolution, bounds=bounds)

        ((kp_low, kp_high), (kz_low, kz_high)) = calculate_kp_kz_bounds(self.arr)
        if "kp" in bounds:
            kp_low, kp_high = bounds["kp"]

        if "kz" in bounds:
            kz_low, kz_high = bounds["kz"]

        inferred_kp_res = (kp_high - kp_low + 2 * K_SPACE_BORDER) / len(self.arr.coords["phi"])
        inferred_kp_res = [b for b in MOMENTUM_BREAKPOINTS if b < inferred_kp_res][-1]

        # go a bit finer here because it would otherwise be very coarse
        inferred_kz_res = (kz_high - kz_low + 2 * K_SPACE_BORDER) / len(self.arr.coords["hv"])
        inferred_kz_res = [b for b in MOMENTUM_BREAKPOINTS if b < inferred_kz_res][-1]

        coordinates["kp"] = np.arange(
            kp_low - K_SPACE_BORDER, kp_high + K_SPACE_BORDER, resolution.get("kp", inferred_kp_res)
        )
        coordinates["kz"] = np.arange(
            kz_low - K_SPACE_BORDER, kz_high + K_SPACE_BORDER, resolution.get("kz", inferred_kz_res)
        )

        base_coords = {k: v for k, v in self.arr.coords.items() if k not in ["eV", "phi", "hv"]}

        coordinates.update(base_coords)

        return coordinates

    def kspace_to_hv(
        self, binding_energy: np.ndarray, kp: np.ndarray, kz: np.ndarray, *args: Any, **kwargs: Any
    ) -> np.ndarray:
        """Converts from momentum back to the raw photon energy."""
        if self.hv is None:
            inner_v = self.arr.S.inner_potential
            wf = self.arr.S.work_function

            is_constant_shift = True
            if not isinstance(binding_energy, np.ndarray):
                is_constant_shift = True
                binding_energy = np.array([binding_energy])

            self.hv = np.zeros_like(kp)
            _kspace_to_hv(kp, kz, self.hv, -inner_v - binding_energy + wf, is_constant_shift)

        return self.hv

    def kspace_to_phi(
        self, binding_energy: np.ndarray, kp: np.ndarray, kz: np.ndarray, *args: Any, **kwargs: Any
    ) -> np.ndarray:
        """Converts from momentum back to the hemisphere angle axis."""
        if self.phi is not None:
            return self.phi

        if self.hv is None:
            self.kspace_to_hv(binding_energy, kp, kz, *args, **kwargs)

        kinetic_energy = binding_energy + self.hv - self.arr.S.work_function

        self.phi = np.zeros_like(self.hv)

        _kp_to_polar(
            kinetic_energy,
            kp,
            self.phi,
            self.arr.S.inner_potential,
            self.arr.S.phi_offset,
        )

        try:
            self.phi = self.calibration.correct_detector_angle(eV=binding_energy, phi=self.phi)
        except:
            pass

        return self.phi

    def conversion_for(self, dim: str) -> Callable:
        """Looks up the appropriate momentum-to-angle conversion routine by dimension name."""

        def with_identity(*args, **kwargs):
            return self.identity_transform(dim, *args, **kwargs)

        return {
            "eV": self.kspace_to_BE,
            "hv": self.kspace_to_hv,
            "phi": self.kspace_to_phi,
        }.get(dim, with_identity)
