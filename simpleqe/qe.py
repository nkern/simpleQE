"""
qe.py
-----

A simple quadratic estimator for 21 cm intensity mapping.
"""

import numpy as np

from . import utils


class QE:
    """
    A simple quadratic estimator for 21 cm intensity mapping
    """

    def __init__(self, freqs, x1, x2=None, C=None, spw=None, cosmo=None, Omega_Eff=None):
        """
        A simple quadratic estimator for 21 cm intensity mapping
        
        Parameters
        ----------
        freqs : ndarray (Nfreqs)
            frequency array of x in Hz
        x1 : ndarray (Ntimes, Nfreqs)
             Complex visibility data [milli-Kelvin] as left-hand input for QE
        x2 : ndarray (Ntimes, Nfreqs)
             Complex visibility data as right-hand input for QE
             Default is x1
        C : ndarray (Nfreqs, Nfreqs)
            data covariance, used for errorbars
        spw : tuple or slice object
            Delineates spw channels for power spectrum estimation, of shape (start, stop) channel
            This is used for wideband GPR, where filtering is applied across all freqs
            but pspec estimation is over a subband. Default is the entire band.
        cosmo : Cosmology object
            Adopted cosmology. Default is utils.Cosmology default.
        Omega_Eff : float
            Used for normalizing power spectra.
            Omega_Eff = Omega_p^2 / Omega_pp, where Omega_p is the sky
            integral of the primary beam power [radians^2], and Omega_pp
            is the sky integral of the squared primary beam power [radians^2].
            See HERA Memo #27.
            Default of None results in scalar = 1.0

        Notes
        -----
        The code adopts the following defintions

        c_a^n = e^{-2pi i a n / N}, (Nfreqs, 1)
        Q_a = c_a c_a^t
        uE_a = 0.5 R^t Q_a R
        H_ab = tr(uE_a Q_b)
        q_a = x_1^t uE_a x_2
        M = H^-1/2 or propto I
        E_a = M_ab uE_b
        p_a = M_ab q_b = x_1^t E_a x_2
        W = M H
        V_ab = 2 tr(C E_a C E_b)
        b_a = tr(C E_a)
        """
        # assign data and metadata
        # if x2 is not provided, just use x1
        self.x1 = x1
        if x2 is None:
            self.x2 = x1
        else:
            self.x2 = x2
        self.C = C
        self.freqs = freqs
        self.Nfreqs = len(freqs)

        # spectral window selection
        if spw is None:
            self.spw = slice(None)
        else:
            self.spw = spw
        self.spw_Nfreqs = len(freqs[self.spw])
        self.Nbps = self.spw_Nfreqs  # modified by self.compute_Q
        self.dfreq = np.diff(freqs)[0]

        # cosmology
        if cosmo is None:
            self.cosmo = utils.Cosmology()
        else:
            self.cosmo = cosmo
        self.avg_f = np.mean(freqs)
        self.avg_z = self.cosmo.f2z(self.avg_f)
        self.t2k = self.cosmo.tau_to_kpara(self.avg_z)
        self.X2Y = self.cosmo.X2Y(self.avg_z)

        # power spectrum scalar normalization
        # see HERA Memo #27 and Appendix of Parsons et al. 2014
        # the frequency-dependent taper is handled automatically by R
        # so here B_pp = B_p, and scalar = X2Y * Omega_Eff * B_p
        if Omega_Eff is not None:
            self.scalar = self.X2Y * Omega_Eff * (self.spw_Nfreqs * self.dfreq)
        else:
            self.scalar = 1.0

    def set_R(self, R):
        """
        Set weighting matrix for QE.
        For proper OQE, this should be C^-1.
        For wideband GPR, input R should be square,
        which is then saved as R[self.spw, :]
        
        Parameters
        ----------
        R : ndarray (Nfreqs, Nfreqs)

        Results
        -------
        self.R
        """
        self.R = R[self.spw, :]

    def compute_Q(self, prior=None, bp_thin=None):
        """
        Compute Q = dC / dp

        Parameters
        ----------
        prior : ndarray (Ndelays,)
            Bandpower prior. Re-defines Q^prime_a = prior_a * Q_a
            And defines p^prime_a = p_a / prior_a
        bp_thin : int
            If not None, decimate the band power k values by this amount.
            Eg. bp_dlys = dlys[::bp_thin]
        """
        # compute Q = dC/dp
        # number of band powers is spw_Nfreqs
        if bp_thin is None:
            bp_thin = 1
        Nbps = int(np.ceil(self.spw_Nfreqs / bp_thin))
        self.Nbps = Nbps

        # get DFT vectors, separable components of Q matrix. !! This uses an inverse fft without 1 / N!!
        self.qft = np.fft.ifft(np.eye(self.spw_Nfreqs), axis=-1)[::bp_thin, :] * self.spw_Nfreqs
        self.qft = np.fft.fftshift(self.qft, axes=0)

        # create Nbps x spw_Nfreqs x spw_Nfreqs Q matrix
        self.Q = np.array([_q[None, :].T.conj().dot(_q[None, :]) for _q in self.qft])

        # if R is not square, create a zero-padded Q matrix for computing H_ab = tr[R.T Q_a R Q_zpad_b]
        # if R is square, self.Q_zpad = self.Q
        if self.Nfreqs == self.spw_Nfreqs:
            self.Q_zpad = self.Q
        else:
            self.Q_zpad = np.zeros((Nbps, self.Nfreqs, self.Nfreqs), dtype=np.complex128)
            self.Q_zpad[:, self.spw, self.spw] = self.Q

        # set prior
        if prior is not None:
            self.prior = prior
            for i, p in enumerate(prior):
                self.Q[i] *= p
                self.Q_zpad[i] *= p

        # bandpower k bins
        self.dlys = np.fft.fftshift(np.fft.fftfreq(self.spw_Nfreqs, np.diff(self.freqs)[0]))[::bp_thin] * 1e9
        self.kp = self.dlys * self.t2k / 1e9
        self.kp_mag = np.abs(self.kp)

    def _compute_uE(self, R, Q):
        return 0.5 * np.array([R.T.conj() @ Qa @ R for Qa in Q])

    def _compute_H(self, uE, Q_zpad):
        return np.array([[np.trace(uEa @ Qb) for Qb in Q_zpad] for uEa in uE])

    def compute_H(self, enforce_real=True):
        """
        Compute H_ab = tr[uE_a Q_b]
        Also computes Q and uE.
        For R = C^-1, H = F is the Fisher matrix
        
        Parameters
        ----------
        enforce_real : bool
            If True, take real component of H matrix,
            assuming imaginary component is numerical noise.

        Results
        -------
        self.uE, self.H
        """
        if not hasattr(self, 'R'):
            raise ValueError("No R matrix attached to object")
        if not hasattr(self, 'Q'):
            raise ValueError("Must first run compute_Q")

        # compute un-normed E and then H
        self.uE = self._compute_uE(self.R, self.Q)
        self.H = self._compute_H(self.uE, self.Q_zpad)
        if enforce_real:
            self.H = self.H.real

    def _compute_q(self, x1, x2, uE):
        # this is x1^dagger uE x2, but looks weird due to shape of x1, x2
        return np.array([np.diagonal(x1.conj() @ uEa @ x2.T) for uEa in uE])

    def compute_q(self):
        """
        Compute q: un-normalized band powers
        Must first compute_H
        
        Results
        -------
        self.q
        """
        if not hasattr(self, 'H'):
            raise ValueError("Must first run compute_H")
        self.q = self._compute_q(self.x1, self.x2, self.uE)

    def _compute_M(self, norm, H, pinv=False, rcond=1e-15):
        if norm == 'I':
            Hsum = np.sum(H, axis=1)
            return np.diag(1. / Hsum) * self.scalar
            #return np.diag(np.true_divide(1.0, Fsum, where=~np.isclose(Fsum, 0, atol=1e-15)))
        elif norm == 'H^-1':
            if pinv:
                return np.linalg.pinv(H, rcond=rcond) * self.scalar
            else:
                return np.linalg.inv(H) * self.scalar
        elif norm == 'H^-1/2':
            u,s,v = np.linalg.svd(H)
            truncate = s > (s.max() * rcond)
            u, s, v = u[:, truncate], s[truncate], v[truncate, :]
            M = v.T.conj() @ np.diag(1/np.sqrt(s)) @ u.T.conj()
            W = M @ H
            # normalize
            M /= W.sum(axis=1)[:, None]
            return M * self.scalar
        else:
            raise ValueError("{} not recognized".format(norm))

    def _compute_W(self, M, H):
        return M @ H
    
    def _compute_E(self, M, uE):
        return np.array([np.sum(m[:, None, None] * uE, axis=0) for m in M])

    def _compute_b(self, C, E):
        return np.array([np.trace(C @ Ea) for Ea in E])
    
    def compute_MW(self, norm='I', pinv=False, rcond=1e-15):
        """
        Compute normalization and window functions

        Parameters
        ----------
        norm : str, ['I', 'H^-1', 'H^-1/2']
            Bandpower normalization matrix type

        pinv : bool
            If True use pseudo-inverse in compute_M

        rcond : float
            Relative condition for truncation in pinv or
            svd of norm='H^-1' or norm='H^-1/2'

        Results
        -------
        self.M, self.W, self.uE
        """
        self.norm = norm
        self.kp_mag = np.abs(self.kp)
        # get normalization matrix
        assert hasattr(self, 'H'), "Must first run compute_H"
        self.M = self._compute_M(norm, self.H, pinv=pinv, rcond=rcond)
        # compute window functions
        self.W = self._compute_W(self.M, self.H) / self.scalar
        # compute normalized E matrix
        self.E = self._compute_E(self.M, self.uE)

    def _compute_p(self, M, q):
        return M @ q

    def compute_p(self, C_bias=None):
        """
        Compute normalized bandpowers and bias term.
        Must first compute_q(), and compute_MW()

        Parameters
        ----------
        C_bias : ndarray (Nfreqs, Nfreqs), optional
            Data covariance for bias term.
            Default is no bias term.

        Results
        -------
        self.p, self.b
        """
        self.kp_mag = np.abs(self.kp)
        # compute normalized bandpowers
        assert hasattr(self, 'q'), "Must first run compute_q()"
        assert hasattr(self, "M"), "Must first run compute_MW()"
        self.p = self._compute_p(self.M, self.q)
        # compute bias term
        if C_bias is not None:
            self.b = self._compute_b(C_bias, self.E)[:, None]
        else:
            self.b = np.zeros_like(self.p)

    def _compute_V(self, C, E):
        # compute C @ Ea once
        CE = [C @ Ea for Ea in E]
        # compute all cross terms
        return 2 * np.array([[np.trace(CEa @ CEb) for CEb in CE] for CEa in CE])

    def compute_V(self, C_data=None):
        """
        Compute bandpower covariance.
        Must run compute_MW() first.

        Parameters
        ----------
        C_data : ndarray (Nfreqs, Nfreqs), optional
            Data covariance for errorbar estimation.
            Default is self.C

        Results
        -------
        self.V
        """
        # compute bandpower covariance
        assert hasattr(self, 'E'), "Must first run compute_MW()"
        if C_data is None:
            self.V = np.eye(self.Nbps, dtype=np.complex)
        else:
            self.V = self._compute_V(C_data, self.E)

    def spherical_average(self, kp_sph=None):
        """
        Spherical average onto |k| axis

        Parameters
        ----------
        kp_sph : ndarray of k values. Default is |self.kp|

        p_cyl = A p_sph
        p_sph = [A.T C_cyl^-1 A]^-1 A.T C_cyl^-1 p_cyl
        C_sph = [A.T C_cyl^-1 A]^-1
        W_sph = [A.T C_cyl^-1 A]^-1 A.T C_cyl^-1 W_cyl A
        """
        # identity weighting if no errors
        if not hasattr(self, 'V'):
            self.V = np.eye(self.p.shape[0])

        if kp_sph is None:
            self.kp_sph = np.unique(np.abs(self.kp))
        else:
            self.kp_sph = kp_sph

        # construct A matrix
        A = np.zeros((len(self.kp), len(self.kp_sph)))
        for i, k in enumerate(self.kp):
            A[i, np.argmin(np.abs(self.kp_sph - np.abs(k)))] = 1.0

        # get p_sph
        Vinv = np.linalg.inv(self.V)
        AtVinv = A.T @ Vinv
        AtVinvAinv = np.linalg.inv(AtVinv @ A)
        self.p_sph = AtVinvAinv @ AtVinv @ self.p
        self.b_sph = AtVinvAinv @ AtVinv @ self.b
        
        # get V_sph
        self.V_sph = AtVinvAinv

        # get W_sph
        self.W_sph = AtVinvAinv @ AtVinv @ self.W @ A

    def compute_MWVp(self, norm='I', C_data=None, C_bias=None, sph_avg=True, kp_sph=None):
        """
        Shallow wrapper for compute_MW, p, V and spherical average

        Parameters
        ----------
        norm : see compute_MW()
        C_data : see compute_V()
        C_bias : see compute_p()
        sph_avg : If True, run self.spherical_average()
        kp_sph : see spherical_average()
        """
        self.compute_MW(norm=norm)
        self.compute_p(C_bias=C_bias)
        self.compute_V(C_data=C_data)
        if sph_avg:
            self.spherical_average(kp_sph=kp_sph)

    def _compute_dsq(self, kp, p, b, V):
        kfac = kp[:, None]**3 / 2 / np.pi**2
        dsq_p = p * kfac
        dsq_b = b * kfac
        if V is not None:
            Ik = np.diag(kfac.squeeze())
            dsq_V = Ik @ V @ Ik
        else:
            dsq_V = None
        return dsq_p, dsq_b, dsq_V

    def compute_dsq(self):
        """
        Compute Delta-Square from self.kp_sph

        Result
        -----
        self.dsq
        self.dsq_b
        self.dsq_V
        """
        assert hasattr(self, 'p_sph')
        self.dsq, self.dsq_b, self.dsq_V = self._compute_dsq(self.kp_sph, self.p_sph, self.b_sph, self.V_sph)

