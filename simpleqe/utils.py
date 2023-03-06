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
    def __init__(self, H0=67.7, Om0=0.3075, Ob0=0.0486):
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
        """
        super().__init__(H0, Om0, Tcmb0=2.725, Ob0=Ob0, Neff=3.05, m_nu=[0., 0., 0.06] * units.eV)

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
        return 2 * np.pi / self.dRpara_df(z)


def gen_data(freqs, Kfg, Keor, Knoise, Ntimes=1, fg_mult=1, eor_mult=1, noise_mult=1,
             ind_noise=True, data_spw=None, pspec_spw=None, seed=0, cosmo=None, Omega_Eff=None):
    """Generate mock dataset and return DelayQE objects

    Parameters
    ----------
    freqs : ndarray
        Frequency array in Hz
    Kfg : callable
        Input freqs[:, None], output foreground cov ndarray
    Kfg : callable
        Input freqs[:, None], output EoR cov ndarray
    Knoise : callable
        Input freqs[:, None], output noise cov ndarray
    Ntimes : int
        Number of times. Each sample is an independent draw.
    fg_mult : float
        Multiplier of foreground covariance given input
    eor_mult : float
        Multiplier of eor covariance given input
    noise_mult : float
        Multiplier of noise covariance given input
    ind_noise : bool
        If True, draw two independent noise realizations and
        assign as x1 and x2, else x2 is a copy of x1.
    data_spw : tuple or slice object
        Sets spectral window of data from input freqs
    pspec_spw : tupe or slice object
        Sets spectral window of pspec estimation from input freqs
    seed : int
        Random seed to set before drawing data
    cosmo : Cosmology object
        Default is utils.Cosmology() default.
    Omega_Eff : float
        Effective primary beam area, see HERA Memo #27

    Returns
    -------
    DelayQE object
        Full dataset
    DelayQE object
        Foreground dataset
    DelayQE object
        EoR dataset
    DelayQE object
        Noise dataset
    """
    from simpleqe.qe import DelayQE
    if data_spw is None:
        data_spw = slice(None)
    Kf = Kfg(freqs[:, None]) * fg_mult
    Ke = Keor(freqs[:, None]) * eor_mult
    Kn = Knoise(freqs[:, None]) * noise_mult

    if cosmo is None:
        cosmo = Cosmology()
    if Omega_Eff is None:
        Omega_Eff = 1

    np.random.seed(seed)
    mean = np.zeros_like(freqs)
    f = np.atleast_2d(mn.rvs(mean, Kf/2, Ntimes) + 1j * mn.rvs(mean, Kf/2, Ntimes)).T[None, data_spw]
    e = np.atleast_2d(mn.rvs(mean, Ke/2, Ntimes) + 1j * mn.rvs(mean, Ke/2, Ntimes)).T[None, data_spw]
    n1 = np.atleast_2d(mn.rvs(mean, Kn/2, Ntimes) + 1j * mn.rvs(mean, Kn/2, Ntimes)).T[None, data_spw]
    x1 = f + e + n1
    if ind_noise:
        n2 = np.atleast_2d(mn.rvs(mean, Kn/2, Ntimes) + 1j * mn.rvs(mean, Kn/2, Ntimes)).T[None, data_spw]
        x2 = f + e + n2
    else:
        x2 = x1.copy()
    
    # metadata
    df = freqs[1] - freqs[0]
    dx = df * cosmo.dRpara_df(cosmo.f2z(freqs.mean()))
    kperp = [0.0]  # assume this is kperp of 0 even though this isn't the auto-correlation

    # compute cosmological scalar
    spw = pspec_spw if pspec_spw is not None else slice(None)
    scalar = cosmo.X2Y(cosmo.f2z(freqs.mean())) * Omega_Eff * df # Nfreqs is omitted b/c FT uses ortho convention

    D = DelayQE(x1, dx, kperp, x2=x2, C=(Kf + Ke + Kn)[data_spw, data_spw], idx=pspec_spw, scalar=scalar)
    F = DelayQE(f,  dx, kperp, C=(Kf)[data_spw, data_spw], idx=pspec_spw, scalar=scalar)
    E = DelayQE(e,  dx, kperp, C=(Ke)[data_spw, data_spw], idx=pspec_spw, scalar=scalar)
    N = DelayQE(n1, dx, kperp, x2=n2, C=(Kn)[data_spw, data_spw], idx=pspec_spw, scalar=scalar)

    return D, F, E, N


def ravel_mats(mat1, mat2, cov=False):
    """
    Given two square matrices mat1 n x n and mat2 m x m,
    ravel and multiply them and return a matrix nm x nm.
    This is consistent with their diagonals representing two
    dimensions of an array of shape (n, m) and calling np.ravel(arr).
    mat1 or mat2 can also be fed as a vector (assumed to be diagonal of matrix),
    or an integer (assumed to be identity matrix of integer length).

    Parameters
    ----------
    mat1 : ndarray or int
        First matrix to ravel. If this is a diagonal matrix,
        this can be sped-up by feeding mat1.diagonal().
        If an integer is fed, this becomes np.eye(mat1) and
        simple broadcasting is applied.
    mat2 : ndarray or int
        Second matrix to ravel. If this is a diagonal matrix,
        this can be sped-up by feeding mat2.diagonal().
        If an integer is fed, this becomes np.eye(mat1) and
        simple broadcasting is applied.
    cov : bool, optional
        If True, mat1 and mat2 represent covariance matrices,
        which require special normalization. Otherwise, keep
        this set to False (e.g. window functions, FT operators)

    Returns
    -------
    ndarray
    """
    # if one of mat1 or mat2 is fed as an integer
    # then we are simply broadcasting along this dimension
    identity = False
    if isinstance(mat1, int):
        identity = True
        assert isinstance(mat2, np.ndarray)
        if mat2.ndim == 1:
            mat1 = np.ones(mat1)
        elif mat2.ndim == 2:
            mat1 = np.eye(mat1)
    elif isinstance(mat2, int):
        identity = True
        assert isinstance(mat1, np.ndarray)
        if mat1.ndim == 1:
            mat2 = np.ones(mat2)
        elif mat1.ndim == 2:
            mat2 = np.eye(mat2)

    # if either mat1 or mat2 is ndim=2, make sure both are
    if mat1.ndim != mat2.ndim:
        if mat1.ndim == 1:
            mat1 = np.diag(mat1)
        else:
            mat2 = np.diag(mat2)

    # take kronecker product
    out = np.kron(mat1, mat2)

    if cov and not identity:
        # normalize units if mat1 and mat2 are covariances
        if out.ndim == 1:
            # if just variances, use geometric mean
            out = np.sqrt(out)
        else:
            # if off-diagonals, normalize by diagonal
            T = 1 / out.diagonal()**(1./4)
            out = T[:, None] * out * T[None, :]

    return out


def interp_Wcdf(W, k, lower_perc=0.16, upper_perc=0.84):
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
        dk of WF's 16th (default) percentile from median
    ndarray
        dk of WF's 84th (default) percentile from median
    """
    # get cdf: take sum of only abs(W)
    W = np.abs(W) / W.sum(axis=1, keepdims=True)
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
        low_err.append(m - interp(lower_perc))
        hi_err.append(interp(upper_perc) - m)

    return np.array(med), np.array(low_err), np.array(hi_err)


def gauss_cov(freqs, ell, var=1.0):
    """
    Gaussian covariance

    Parameters
    ----------
    freqs : array-like, (Nfreqs,)
    ell : float, length scale in units of freqs
    var : float, variance
    """
    f = np.atleast_2d(freqs)
    return var * np.exp(-0.5 * (f - f.T)**2 / ell**2)


def exp_cov(freqs, ell, var=1.0):
    """
    Exponential covariance

    Parameters
    ----------
    freqs : array-like, (Nfreqs,)
    ell : float, length scale in units of freqs
    var : float, variance
    """
    f = np.atleast_2d(freqs)
    return var * np.exp(-np.abs(f-f.T) / ell)


def diag_cov(freqs, var=1.0):
    """
    Diagonal covariance

    Parameters
    ----------
    freqs : array-like, (Nfreqs,)
    diag : float or array-like, variance of diagonal
    """
    return np.eye(len(freqs)) * var
