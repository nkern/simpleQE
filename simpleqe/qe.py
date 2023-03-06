"""
qe.py
-----

A simple quadratic estimator for 21 cm intensity mapping.
"""

import numpy as np
from . import utils


class QE:
    """
    A simple quadratic estimator, assuming that the covariance
    derivative Q = dC/dp is separable along each data dimension
    (i.e. the data is a uniform, rectangular grid).
    """
    def __init__(self, x1, dx, x2=None, idx=None, scalar=None, C=None, useQ=False):
        """        
        Parameters
        ----------
        x1 : ndarray (Npix1, Npix2, ..., Nother)
            Data vector for LHS of QE, with Ndim data dimensions (Npix1, Npix2, ...)
            and 1 final dimension to broadcast over (Nother,). E.g. (Nfreqs, 1)
            for a visibility based estimator.
        dx : list
            Delta-x units for each Ndim data dimension in x1.
            The Fourier convention is defined as 2pi/dx.
        x2 : ndarray (Npix1, Npix2, ..., Nother), optional
             Data vector for RHS of QE, with same convention as x1. Default is
             to use x1.
        idx : list of tuple or slice objects, optional
            List of indexing objects for each data dimension of x1.
            This allows you to specify a subset of pixels in each dimension
            specifically for power spectrum estimation (i.e. a spectral window).
            A wider bandwidth can be useful for specialized data weighting
            (e.g. inverse covariance), while still estimating the pspec
            over a narrower bandwidth.
        scalar : float, optional
            Overall normalization for power spectra.
            E.g. for delay spectrum (see HERA Memo #27)
            this is X2Y * Omega_Eff * Nfreqs * dfreq
            assuming unit-normalized qft matrices
            (i.e. np.fft.fft convention).
            Default is 1.
        C : list, optional
            List of covariance matrices for each
            data dimension in x1. Only for metadata purposes.
            Note that if feeding x1 and x2 with indepenent noise
            realizations, one should divide C by sqrt(2).
        useQ : bool, optional
            If True, form outerproduct Q_a = qft_a qft_a^T
            and compute downstream products as well,
            otherwise use qft_a approximation instead (default).

        Notes
        -----
        The code adopts the following defintions

        qft_a = e^{-2pi i a n / N}
        Q_a = qft_a qft_a^T
        qR_a = qft_a R
        RQR_a = R^T Q_a R
        H_ab = tr[qR_a^T qR_b] or
        H_ab = tr[R^T Q_a R Q_b]
        q_a = x1^T qR_a^T qR_a x2 or
        q_a = x1^T R^T Q_a R x2
        M = H^-1 or H^-1/2 or propto I
        p_a = M_ab q_b
        W = M H
        E_a = M_ab (qR_b^T qR_b) or
        E_a = M_ab R^T Q_b R
        V_ab = tr(C E_a C E_b) (variance of p.real)
        b_a = tr(C E_a)
        """
        # assign data
        if x1.ndim == 1:
            x1 = x1.reshape(-1, 1)
        self.x1 = x1
        self.Ndim = x1.ndim - 1
        # if x2 is not provided, just use x1
        if x2 is not None:
            if x2.ndim == 1:
                x2 = x2.reshape(-1, 1)
        self.x2 = x2

        # assign metadata
        assert len(dx) == self.Ndim, "x1 must be of shape Ndim+1, dx of shape Ndim"
        self.dx = dx
        self.idx = idx
        self.scalar = scalar if scalar is not None else 1
        self.C = C
        self.useQ = useQ

    def set_R(self, R=None):
        """
        Set data weighting matrix.

        Parameters
        ----------
        R : list of 2d arrays, optional
            A list of length self.Ndim, holding
            a matrix of shape (Npix_out, Npix_in)
            for each data dimension in x1.
            Default is identity weighting.

        Results
        -------
        self.R
        """
        if R is None:
            R = [np.eye(self.x1.shape[i]) for i in range(self.Ndim)]
        if self.idx is not None:
            R = [r[idx] if idx is not None else r for r, idx in zip(R, self.idx)]

        self.R = R

    def compute_qft(self):
        """
        Compute the column vector qft, which goes into
        the outer product Q_a = qft_a qft_a^T for each
        data dimension in self.x1

        Note that we compute qft of shape (Nbp, Npix)
        and we also compute a qft_pad of shape (Nbp, Npix+Npad)
        where Npix is set by self.idx (i.e. indices over which
        we estimate the power spectrum) and Npix+Npad are additional
        (optional) pixels used in the R weighting step. qft_pad
        is only needed if self.idx is used to select a subset
        of pixels.
        """
        assert hasattr(self, 'R'), "Must first run set_R()"
        # compute k-modes for each data dimension
        shape = [R.shape for R in self.R]
        self.k = [np.fft.fftshift(2*np.pi*np.fft.fftfreq(n[0], dx)) for n, dx in zip(shape, self.dx)]

        # compute qft for each data dimension: this is inverse ft without 1 / N
        self.qft = [np.fft.fftshift(np.fft.ifft(np.eye(n[0])*n[0]), axes=0) for n in shape]
        self._compute_qft_pad()

        # compute Q
        if self.useQ:
            self._compute_Q()

        # compute qft dotted into R
        self._compute_qR()

    def _compute_qft_pad(self):
        assert hasattr(self, 'R'), "Must first run set_R()"
        self.qft_pad = []
        shape = [R.shape for R in self.R]
        for q, n in zip(self.qft, shape):
            p = np.zeros((n[0], (n[1]-n[0])//2), dtype=float)
            self.qft_pad.append(np.hstack([p, np.hstack([q, p])]))

    def _compute_Q(self):
        # compute outer products
        self.Q = [np.einsum('ai,aj->aij', qft.conj(), qft) for qft in self.qft]
        self.Q_pad = [np.einsum('ai,aj->aij', qft.conj(), qft) for qft in self.qft_pad]
        ## TODO: apply bandpower sinc matrix correction
        ## TODO: expose bandpower prior

    def _compute_qR(self):
        """
        Compute dot product of weighting matri (R)
        and FT matrices (qft or Q)
        """
        assert hasattr(self, 'R')
        assert hasattr(self, 'qft')
        # compute qft dotted into R
        self.qR = [q @ r for q, r in zip(self.qft, self.R)]
        if self.useQ:
            self.RQR = [r.T.conj() @ q @ r for q, r in zip(self.Q, self.R)]

    def compute_q(self):
        """
        Compute q: un-normalized band powers
        
        Results
        -------
        self.q
        """
        assert hasattr(self, 'qft'), "must run compute_qft()"
        x1, x2 = self.x1, self.x2

        # go the Q route: dot RQR into x1, and then x2 into x1Q
        if self.useQ:
            # dot Qs into x1
            x1_str = 'ijkl'[:self.Ndim]
            bp_strs = ['ap', 'bq', 'cr', 'ds'][:self.Ndim]
            x1 = x1.conj()
            in_str = x1_str
            for i, Q in enumerate(self.RQR):
                Q_str = bp_strs[i][0] + x1_str[i] + bp_strs[i][1]
                out_str = in_str.replace(x1_str[i], bp_strs[i])
                x1 = np.einsum('{}...,{}->{}...'.format(in_str, Q_str, out_str), x1, Q)
                in_str = out_str

            # dot x2 into x1_Q
            x2_str = ''.join(bp[1] for bp in bp_strs)
            out_str = ''.join(bp[0] for bp in bp_strs)
            if x2 is None:
                x2 = self.x1
            q = np.einsum('{}...,{}...->{}...'.format(in_str, x2_str, out_str), x1, x2)

        # go the qR route: dot qR into x1 and x2, then multiply
        else:
            es_str = 'ijkl'[:self.Ndim]
            for i, qr in enumerate(self.qR):
                str_in = es_str.replace(es_str[i], 'b')
                str_out = es_str.replace(es_str[i], 'a')
                x1 = np.einsum("ab,{}...->{}...".format(str_in, str_out), qr, x1)
                if x2 is not None:
                    x2 = np.einsum("ab,{}...->{}...".format(str_in, str_out), qr, x2)

            if x2 is None:
                x2 = x1

            q = x1.conj() * x2

        self.q = q

    def compute_H(self):
        """
        Compute H_ab = tr[R^T Q_a R Q_b]
        For R = C^-1, H = F is the Fisher matrix

        Results
        -------
        self.H
        """
        assert hasattr(self, 'qft'), "First run compute_qft()"

        H = []
        # Q route
        if self.useQ:
            for rqr, qp in zip(self.RQR, self.Q_pad):
                H.append(np.einsum('aij,bji->ab', rqr, qp).real)

        # qft route
        else:
            for qr, qp in zip(self.qR, self.qft_pad):
                H.append(abs(qr @ qp.conj().T)**2)

        self.H = H

    def _compute_M(self, norm, H, rcond=1e-15, Wnorm=True):
        if norm == 'I':
            M = np.eye(len(H)) / H.sum(axis=1)
        elif norm in ['H^-1', 'H^-1/2']:
            u,s,v = np.linalg.svd(H)
            truncate = np.where(s > (s.max() * rcond))[0]
            u, s = u[:, truncate], s[truncate]
            # we use u @ s @ u.T instead of v.T @ s @ u.T
            # because we want to invert within the
            # left space of H, not re-project back to
            # right space of H. This only makes a difference
            # if H is non-square.
            if norm == 'H^-1':
                M = v.T.conj() @ np.diag(1/s) @ u.T.conj()
            elif norm == 'H^-1/2':
                M = v.T.conj() @ np.diag(1/np.sqrt(s)) @ u.T.conj() * np.sqrt(s).sum() / s.sum()
        else:
            raise ValueError("{} not recognized".format(norm))

        if Wnorm:
            # get window functions
            W = M @ H

            # ensure they are normalized
            M /= W.sum(axis=1, keepdims=True).clip(1e-10)

        return M

    def _compute_W(self, M, H):
        return M @ H

    def compute_MW(self, norm='I', rcond=1e-15, Wnorm=True):
        """
        Compute normalization and window functions.
        For H^-1 and H^-1/2, uses SVD pseudoinverse.

        Parameters
        ----------
        norm : str, ['I', 'H^-1', 'H^-1/2']
            Bandpower normalization matrix type
        rcond : float
            Relative condition for truncation
            of svd for norm='H^-1' or norm='H^-1/2'
        Wnorm : bool, optional
            If True, explicitely normalize W such that
            rows sum to unity. This is generally done
            implicitly in each case, but can be slightly
            off depending on choice of norm.

        Results
        -------
        self.M, self.W, self.E
        """
        self.norm = norm
        # get normalization matrix
        assert hasattr(self, 'H'), "Must first run compute_H"
        self.M = [self._compute_M(norm, H, rcond=rcond, Wnorm=Wnorm) for H in self.H]
        if self.useQ:
            self.E = [np.einsum("ab,bij->aij", m, rqr) for m, rqr in zip(self.M, self.RQR)]

        # compute window functions
        self.W = [self._compute_W(M, H) for M, H in zip(self.M, self.H)]

    def compute_p(self, C_bias=None, rcond=1e-15):
        """
        Compute normalized bandpowers and bias term.
        Must first compute_q(), and compute_MW()

        Parameters
        ----------
        C_bias : list of ndarray, optional
            Bias covariance for each data dimension
            in x1. Default is zero bias.
        rcond : float
            Relative condition when decomposing C
            via svd

        Results
        -------
        self.p, self.b
        """
        # compute normalized bandpowers
        assert hasattr(self, 'q'), "Must first run compute_q()"
        assert hasattr(self, "M"), "Must first run compute_MW()"

        # compute p
        es_str = 'abcd'[:self.Ndim]
        p = self.q
        for i, m in enumerate(self.M):
            str_in = es_str.replace(es_str[i], 'j')
            str_out = es_str.replace(es_str[i], 'i')
            p = np.einsum("ij,{}...->{}...".format(str_in, str_out), m, p)

        self.p = p * self.scalar

        # compute bias term
        b = np.zeros(self.p.shape, dtype=float)
        QR = self.E if self.useQ else self.qR
        if C_bias is not None:
            for i, (c, m, qr) in enumerate(zip(C_bias, self.M, QR)):
                if c is not None:
                    # compute normalized bias: tr[E @ C]
                    if self.useQ:
                        nb = np.einsum('aij,ji->a', qr, c)

                    # compute un-normalized bias, then normalize
                    else:
                        # decompose matrix c into A A^T
                        u, s, v = np.linalg.svd(c, hermitian=True)
                        keep = s > s.max() * rcond
                        A = u[:, keep] @ np.diag(np.sqrt(s[keep]))

                        # dot A into qR
                        qrA = qr @ A

                        # take abs sum to get un-normalized bias
                        ub = (abs(qrA)**2).sum(-1)

                        # normalize
                        nb = m @ ub

                    # sum bias terms
                    bshape = [1 for j in range(b.ndim)]
                    bshape[i] = -1
                    b += nb.real.reshape(bshape)

        self.b = b * self.scalar

        if hasattr(self, 'p_avg'):
            delattr(self, 'p_avg')
            delattr(self, 'b_avg')
            delattr(self, 'V_avg')
            delattr(self, 'W_avg')

    def compute_V(self, C=None, diag=True):
        """
        Compute bandpower covariance.
        Must run compute_MW() first.

        Parameters
        ----------
        C : list of ndarray
            List holding a 2D covariance matrix of shape
            (Npix, Npix) for each data dimension in x1.
        diag : bool, optional
            If True, only compute diagonal of
            bandpower covariance. Otherwise compute
            off-diagonal as well.

        Results
        -------
        self.V
        """
        assert hasattr(self, 'M'), "Must run compute_MW()"
        V = []
        if C is None:
            C = [None for m in M]
        qR = self.E if self.useQ else self.qR
        for i, (c, m, qr) in enumerate(zip(C, self.M, qR)):
            # default is None
            v = None

            # update if covariance is provided
            if c is not None:
                # dot E into covariance
                if self.useQ:
                    ec = np.einsum('aij,jk->aik', qr, c)

                # compute E, then dot into covariance
                else:
                    # compute un-normalized E: qR outer product
                    ue = np.einsum("ij,ik->ijk", qr, qr.conj())

                    # compute normalized E
                    e = np.einsum("ij,jkl->ikl", m, ue)

                    # dot into covariance
                    ec = np.einsum("ijk,kl->ijl", e, c)

                if diag:
                    # compute just variance
                    v = np.einsum("aij,aji->a", ec, ec).real

                else:
                    # compute full covariance
                    v = np.einsum("aij,bji->ab", ec, ec).real

                v *= self.scalar**2

            V.append(v)

        self.V = V

    def average_bandpowers(self, k_avg=None, two_dim=True, axis=None):
        """
        Average the computed normalized bandpowers
        and their associated metadata (e.g. covariances
        and window functions).

        Parameters
        ----------
        k_avg : ndarray
            mag(k) to average onto.
        two_dim : bool, optional
            If True, average first two dimensions of self.p,
            otherwise average first three dimensions of self.p.
        axis : int, optional
            If None, perform average of first N dimensions.
            If provided, only average this axis onto k_avg
            (e.g. this is used for folding the power spectra).

        Notes
        -----
        p_xyz = A p_avg
        p_avg = [A.T C_xyz^-1 A]^-1 A.T C_xyz^-1 p_xyz
        C_avg = [A.T C_xyz^-1 A]^-1
        W_avg = [A.T C_xyz^-1 A]^-1 A.T C_xyz^-1 W_xyz A
        """
        assert hasattr(self, 'p'), "Must run self.compute_p()"
        assert hasattr(self, 'V'), "Must run self.compute_V()"

        # get initial arrays
        pi = self.p if not hasattr(self, 'p_avg') else self.p_avg
        bi = self.b if not hasattr(self, 'p_avg') else self.b_avg
        Vi = self.V if not hasattr(self, 'p_avg') else self.V_avg
        Wi = self.W if not hasattr(self, 'p_avg') else self.W_avg
        ki = self.k if not hasattr(self, 'p_avg') else self.k_avg
        s = pi.shape
        if axis is None:
            if two_dim:
                assert pi.ndim > 2, "Cannot 2D average p further"
            else:
                assert pi.ndim > 3, "Cannot 3D average p further"

        if axis is None:
            # averaging first two dimensions: unravel first two dimensions
            if two_dim:
                p = pi.reshape(s[0]*s[1], *s[2:])
                b = bi.reshape(s[0]*s[1], *s[2:])
            else:
                p = pi.reshape(s[0]*s[1]*s[2], *s[3:])
                b = bi.reshape(s[0]*s[1]*s[2], *s[3:])

            # stack matrices to match unraveled dimensions
            if two_dim:
                # stack V
                V1 = Vi[0] if Vi[0] is not None else pi.shape[0] 
                V2 = Vi[1] if Vi[1] is not None else pi.shape[1]
                V = utils.ravel_mats(V1, V2, cov=True)
                # stsack W
                W = utils.ravel_mats(Wi[0], Wi[1])
                # stack k
                K = np.meshgrid(ki[0], ki[1])
                k = np.sqrt(K[0].ravel()**2 + K[1].ravel()**2)

            else:
                # stack V
                V1 = Vi[0] if Vi[0] is not None else pi.shape[0] 
                V2 = Vi[1] if Vi[1] is not None else pi.shape[1]
                V3 = Vi[2] if Vi[2] is not None else pi.shape[2]
                V = utils.ravel_mats(V1, utils.ravel_mats(V2, V3, cov=True), cov=True)
                # stack W
                W = utils.ravel_mats(Wi[0], utils.ravel_mats(Wi[1], Wi[2]))
                # stack k
                K = np.meshgrid(ki[0], ki[1], ki[2])
                k = np.sqrt(K[0].ravel()**2 + K[1].ravel()**2 + K[2].ravel()**2)

            # get k_avg points
            if k_avg is None:
                k_avg = np.linspace(k.min(), k.max(), max(len(ki[0]), len(ki[1]))//2)

        else:
            # otherwise, we are folding this axis
            assert axis < pi.ndim
            p = np.moveaxis(pi, axis, 0)
            b = np.moveaxis(bi, axis, 0)
            V = Vi[axis]
            V = V if V is not None else np.ones(pi.shape[axis])
            W = Wi[axis]
            k = np.abs(ki[axis])

            if k_avg is None:
                k_avg = np.unique(k)

        # construct A matrix: p_xyz = A @ p_avg
        A = np.zeros((len(k), len(k_avg)), dtype=float)
        for i, _k in enumerate(k):
            # nearest neighbor interpolation
            nn = np.argmin(np.abs(k_avg - abs(_k)))
            A[i, nn] = 1.0

            # linear interpolation: not as stable
            #nn = np.sort(np.argsort(np.abs(k_avg - _k))[:2])
            #A[i, nn[0]] = (k_avg[nn[1]] - _k) / (k_avg[nn[1]] - k_avg[nn[0]])
            #A[i, nn[1]] = (_k - k_avg[nn[0]]) / (k_avg[nn[1]] - k_avg[nn[0]])

        # compute inverse matrices
        if V.ndim == 1:
            Vinv = np.diag(1 / V.clip(1e-30))
        else:
            Vinv = np.linalg.pinv(V)
        AtVinv = A.T @ Vinv
        AtVinvAinv = np.linalg.pinv(AtVinv @ A)
        D = AtVinvAinv @ AtVinv

        # get averaged quantities
        p_avg = np.einsum("ij,j...->i...", D, p)
        b_avg = np.einsum("ij,j...->i...", D, b)
        V_avg = AtVinvAinv
        W_avg = D @ W @ A

        if axis is None:
            self.p_avg = p_avg
            self.b_avg = b_avg
            if two_dim:
                self.V_avg = [V_avg] + Vi[2:]
                self.W_avg = [W_avg] + Wi[2:]
                self.k_avg = [k_avg] + ki[2:]
            else:
                self.V_avg = [V_avg] + Vi[3:]
                self.W_avg = [W_avg] + Wi[3:]
                self.k_avg = [k_avg] + ki[3:]

        else:
            self.p_avg = np.moveaxis(p_avg, 0, axis)
            self.b_avg = np.moveaxis(b_avg, 0, axis)
            self.V_avg = Vi[:axis] + [V_avg] + Vi[axis+1:]
            self.W_avg = Wi[:axis] + [W_avg] + Wi[axis+1:]
            self.k_avg = ki[:axis] + [k_avg] + ki[axis+1:]

    def fold_bandpowers(self):
        """
        Average negative and positive k modes for each
        data dimension in self.p.

        Notes
        -----
        self.p_avg : folded bandpowers
        self.b_avg : folded bandpower biases
        self.V_avg : folded bandpower covariances
        self.W_avg : folded bandpower window functions
        self.k_avg : folded bandpower k modes
        """
        assert hasattr(self, 'p'), "Must run self.compute_p()"
        for i in range(self.Ndim):
            self.average_bandpowers(axis=i)

    def compute_MWVp(self, norm='I', rcond=1e-15, C_bias=None, C_errs=None,
                     diag=True, Wnorm=True):
        """
        Shallow wrapper for compute_MW, p, V and spherical average

        Parameters
        ----------
        norm : see compute_MW()
        rcond : see compute_MW()
        C_bias : see compute_p()
        C_errs : see compute_V()
        diag : see compute_V()
        Wnorm : see compute_MW()
        """
        self.compute_MW(norm=norm, rcond=rcond, Wnorm=Wnorm)
        self.compute_p(C_bias=C_bias, rcond=rcond)
        self.compute_V(C=C_errs, diag=diag)

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
        Compute Delta-Square from self.k_avg and self.p_avg

        Result
        -----
        self.dsq
        self.dsq_b
        self.dsq_V
        """
        assert hasattr(self, 'p_avg')
        assert self.p_avg.ndim == 2, "Must average p down to 1D before computing dsq"
        self.dsq, self.dsq_b, self.dsq_V = self._compute_dsq(self.k_avg[0], self.p_avg,
                                                             self.b_avg, self.V_avg[0])


class DelayQE(QE):
    """
    Delay Spectrum QE
    """
    def __init__(self, x1, dx, kperp, x2=None, idx=None, scalar=None, C=None, useQ=False):
        """        
        Parameters
        ----------
        x1 : ndarray (Nbls, Nfreqs, ..., Nother)
            Data vector for LHS of QE, with Ndim data dimensions (Nbls, Nfreqs, ...)
            and 1 final dimension to broadcast over (Nother,)
        dx : float
            Delta-x frequency units [Hz]
        kperp : ndarray
            k_perp values of shape (Nbls,) corresponding to each baseline
            in x1
        x2 : ndarray (Nbls, Nfreqs, ..., Nother)
            Data vector for RHS of QE, with same convention as x1. Default is
            to use x1.
        idx : tuple or slice object, optional
            Indexing for frequency axis.
            A wider bandwidth can be useful for specialized data weighting
            (e.g. inverse covariance), while still estimating the pspec
            over a narrower bandwidth.
        scalar : float, optional
            Overall normalization for power spectra.
            E.g. for delay spectrum see HERA Memo #27.
            Default is 1.
        C : ndarray, optional
            Freq-freq covariance of data. Only used as metadata.
        useQ : bool, optional
            If True, form outerproduct Q_a = qft_a qft_a^T
            and compute downstream products as well,
            otherwise use qft_a approximation instead (default).

        Notes
        -----
        The code adopts the following defintions

        qft_a = e^{-2pi i a n / N}
        qR_a = qft_a R
        H_ab = tr[qR_a^T qR_b]
        q_a = x1^T qR_a^T qR_a x2
        M = H^-1 or H^-1/2 or propto I
        p_a = M_ab q_b
        W = M H
        E_a = M_ab (qR_b^T qR_b)
        V_ab = tr(C E_a C E_b) (variance of p.real)
        b_a = tr(C E_a)
        """
        super().__init__(x1, [1, dx], x2=x2, idx=[slice(None), idx], scalar=scalar, C=C, useQ=useQ)
        self.kperp = kperp

    def set_R(self, R=None):
        """
        Set frequency weighting matrix

        Parameters
        ----------
        R : ndarray
            (Nspw_freqs, Nfreqs) weighting matrix
        """
        super().set_R([np.eye(self.x1.shape[0]), R])

    def compute_qft(self):
        """
        Compute the column vector qft, which goes into
        the outer product Q_a = qft_a qft_a^T for each
        data dimension in self.x1

        Note that we compute qft of shape (Nbp, Npix)
        and we also compute a qft_pad of shape (Nbp, Npix+Npad)
        where Npix is set by self.idx (i.e. indices over which
        we estimate the power spectrum) and Npix+Npad are additional
        (optional) pixels used in the R weighting step. qft_pad
        is only needed if self.idx is used to select a subset
        of pixels.
        """
        super().compute_qft()

        # update objects for delay spectrum estimator
        shape = [R.shape for R in self.R]
        self.k[0] = self.kperp
        self.qft[0] = np.eye(self.x1.shape[0])
        self._compute_qft_pad()
        if self.useQ:
            self._compute_Q()
        self._compute_qR()

    def compute_H(self):
        """
        Compute H_ab = tr[R^T Q_a R Q_b]
        For R = C^-1, H = F is the Fisher matrix.

        Modifies the H[0] matrix based on delay spectrum.

        Results
        -------
        self.H
        """
        super().compute_H()

    def compute_p(self, C_bias=None, rcond=1e-15):
        """
        Compute normalized bandpowers and bias term.
        Must first compute_q(), and compute_MW()

        Parameters
        ----------
        C_bias : ndarray, optional
            Bias covariance for frequency dimension
            in x1. Default is zero bias.
        rcond : float
            Relative condition when decomposing C
            via svd

        Results
        -------
        self.p, self.b
        """
        super().compute_p(C_bias=[None, C_bias], rcond=rcond)

    def compute_V(self, C=None, diag=True):
        """
        Compute bandpower covariance.
        Must run compute_MW() first.

        Parameters
        ----------
        C : ndarray
            2D covariance matrix of shape
            (Nfreqs, Nfreqs) for freq dimension in x1.
        diag : bool, optional
            If True, only compute diagonal of
            bandpower covariance. Otherwise compute
            off-diagonal as well.

        Results
        -------
        self.V
        """
        super().compute_V(C=[None, C], diag=diag)

