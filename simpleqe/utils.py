"""
utils.py
--------

Utility functios for simpleqe
"""

import numpy as np
from scipy.stats import multivariate_normal as mn
from scipy import interpolate
from astropy import units, constants
from astropy.cosmology import FlatLambdaCDM


class Cosmology(FlatLambdaCDM):
    """
    Subclass of astropy.FlatLambdaCDM, with additional methods for 21cm intensity mapping.
    """

    def __init__(self, H0=67.7, Om0=0.3075, Ob0=0.0486, Ode0=0.6910):
        """
        Subclass of astropy.FlatLambdaCDM, with additional methods for 21cm intensity mapping.
        Default parameters are derived from the Planck2015 analysis.

        Parameters
        ----------
        H0 : float
            Hubble parameter at z = 0

        Om0 : float
            Omega matter at z = 0

        Ob0 : float
            Omega baryon at z = 0. Omega CDM is defined relative to Om0 and Ob0.

        Ode0 : float    
            Omega dark energy at z = 0.
        """
        super().__init__(H0, Om0, Ode0, Ob0=Ob0, Tcmb0=2.725, Neff=3.05, m_nu=[0., 0., 0.06] * units.eV)

        # 21 cm specific quantities
        self._f21 = 1.420405751e9  # frequency of 21cm transition in Hz
        self._w21 = 0.211061140542  # 21cm wavelength in meters

    def H(self, z):
        """
        Hubble parameter at redshift z

        Parameters
        ----------
        z : float
            redshift

        Returns
        -------
        float
            Hubble parameter km / sec / Mpc
        """
        return super().H(z).value

    def f2z(self, freq):
        """
        Convert frequency to redshift for the 21 cm line

        Parameters
        ----------
        freq : float
            frequency in Hz

        Returns
        -------
        float
            redshift
        """
        return self._f21 / freq - 1

    def z2f(self, z):
        """
        Convert redshift to frequency for the 21 cm line.

        Parameters
        ----------
        z : float
            redshift

        Returns
        -------
        float
            frequency in Hzv
        """
        return self._f21 / (z + 1)

    def dRperp_dtheta(self, z):
        """
        Conversion factor from angular size (radian) to transverse
        comoving distance (Mpc) at a specific redshift: [Mpc / radians]

        Parameters
        ----------
        z : float
            redshift

        Returns
        -------
        float
            comoving transverse distance [Mpc]
        """
        return self.comoving_transverse_distance(z).value

    def dRpara_df(self, z):
        """
        Conversion from frequency bandwidth to radial comoving distance at a 
        specific redshift: [Mpc / Hz]

        Parameters
        ----------
        z : float
            redshift

        Returns
        -------
        float
            Radial comoving distance [Mpc]
        """
        return (1 + z)**2.0 / self.H(z) * (constants.c.value / 1e3) / self._f21

    def X2Y(self, z):
        """
        Conversion from radians^2 Hz -> Mpc^3 at a specific redshift

        Parameters
        ----------
        z : float
            redshift
        
        Returns
        -------
        float
            Mpc^3 / (radians^2 Hz)
        """
        return self.dRperp_dtheta(z)**2 \
             * self.dRpara_df(z)

    def bl_to_kperp(self, z):
        """
        Conversion from baseline length [meters] to
        tranverse k_perp wavevector [Mpc^-1]

        Parameters
        ----------
        z : float
            redshift

        Returns
        -------
        float
            Conversion factor [Mpc^-1 / meters]
        """
        # Parsons 2012, Pober 2014, Kohn 2018
        return 2 * np.pi / (self.dRperp_dtheta(z) * (constants.c.value / self.z2f(z)))

    def tau_to_kpara(self, z):
        """
        Conversion from delay [seconds] to line-of-sight k_parallel
        wavevector [Mpc^-1]

        Parameters
        ----------
        z : float
            redshift

        Returns
        -------
        float
            Conversion factor [Mpc^-1 / seconds]
        """
        # Parsons 2012, Pober 2014, Kohn 2018
        return 2 * np.pi / self.dRpara_df(z)


def gen_data(freqs, Kfg, Keor, Knoise, Ntimes=1, ffac=1, efac=1, nfac=1,
             data_spw=None, pspec_spw=None, seed=0, scalar=None):
    """Generate mock dataset

    Parameters
    ----------
    freqs : ndarray
        Frequency array in MHz
    Kfg : callable
        Input freqs, output foreground cov ndarray, (Nfreqs, NFreqs)
    Kfg : callable
        Input freqs, output EoR cov ndarray, (Nfreqs, NFreqs)
    Knoise : callable
        Input freqs, output noise cov ndarray, (Nfreqs, NFreqs)
    Ntimes : int
        Number of times. Each sample is an independent draw.
    fg_mult : float
        Multiplier of foreground covariance given input
    eor_mult : float
        Multiplier of eor covariance given input
    noise_mult : float
        Multiplier of noise covariance given input
    data_spw : tuple or slice object
        Sets spectral window of data from input freqs
    pspec_spw : tupe or slice object
        Sets spectral window of pspec estimation from input freqs
    seed : int
        Random seed to set before drawing data

    Returns
    -------
    QE object
        Full dataset
    QE object
        Foreground dataset
    QE object
        EoR dataset
    QE object
        Noise dataset
    """
    from simpleqe import QE
    if data_spw is None:
        data_spw = slice(None)
    Kf = Kfg(freqs[:, None]) * ffac
    Ke = Keor(freqs[:, None]) * efac
    Kn = Knoise(freqs[:, None]) * nfac
    
    np.random.seed(seed)
    mean = np.zeros_like(freqs)
    f = np.atleast_2d(mn.rvs(mean, Kf/2, Ntimes) + 1j * mn.rvs(mean, Kf/2, Ntimes))[:, data_spw]
    e = np.atleast_2d(mn.rvs(mean, Ke/2, Ntimes) + 1j * mn.rvs(mean, Ke/2, Ntimes))[:, data_spw]
    n1 = np.atleast_2d(mn.rvs(mean, Kn/2, Ntimes) + 1j * mn.rvs(mean, Kn/2, Ntimes))[:, data_spw]
    n2 = np.atleast_2d(mn.rvs(mean, Kn/2, Ntimes) + 1j * mn.rvs(mean, Kn/2, Ntimes))[:, data_spw]
    x1 = f + e + n1
    x2 = f + e + n2
    
    D = qe.QE(freqs[data_spw], x1, x2=x2, C=(Kf + Ke + Kn)[data_spw, data_spw], spw=pspec_spw)
    F = qe.QE(freqs[data_spw], f, C=Kf[data_spw, data_spw], spw=pspec_spw)
    E = qe.QE(freqs[data_spw], e, C=Ke[data_spw, data_spw], spw=pspec_spw)
    N = qe.QE(freqs[data_spw], n1, x2=n2, C=Kn[data_spw, data_spw], spw=pspec_spw)
    
    return D, F, E, N


def interp_Wcdf(W, k):
    """
    Construct CDF from normalized window function and interpolate
    to get k at window func's 16, 50 & 84 percentile.

    Parameters
    ----------
    W : ndarray
        Normalized window function of shape (Nbandpowers, Nk)
    k : ndarray
        vector of k modes of shape (Nk,)

    Returns
    -------
    ndarray
        k of WF's 50th percentile
    ndarray
        k of WF's 16th percentile
    ndarray
        k of WF's 84th percentile
    """
    # get cdf: take sum of only abs(W)
    W = np.abs(W)
    Wcdf = np.array([np.sum(W[:, :i+1].real, axis=1) for i in range(W.shape[1]-1)]).T
    
    # get shifted k such that a symmetric window has 50th perc at max value
    kshift = k[:-1] + np.diff(k) / 2
    
    # interpolate each mode (xvalues are different for each mode!)
    med, low_err, hi_err = [], [], []
    for i, w in enumerate(Wcdf):
        interp = interpolate.interp1d(w, kshift, kind='linear', fill_value='extrapolate')
        m = interp(0.5)  # 50th percentile
        #m = k[np.argmax(W[i])]  # mode
        med.append(m)
        low_err.append(m - interp(0.16))
        hi_err.append(interp(0.84) - m)

    return np.array(med), np.array(low_err), np.array(hi_err)
