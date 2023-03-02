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
    derivative Q = dC/dp is separable into the outer product Q = c c^T
    and assuming each data dimension is separable (i.e. on a uniform,
    rectangular grid).
    """
    def __init__(self, x1, dx, x2=None, idx=None, scalar=None):
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
            E.g. for delay spectrum see HERA Memo #27.
            Default is 1.
        Notes
        -----
        The code adopts the following defintions

        qft_a = e^{-2pi i a n / N}
        qR_a = qft_a R
        H_ab = 1/2 tr[qR_a^T qR_b]
        q_a = 1/2 x1^T qR_a^T qR_a x2
        M = H^-1 or H^-1/2 or propto I
        p_a = M_ab q_b
        W = M H
        E_a = 1/2 M_ab (qR_b^T qR_b)
        V_ab = 2 tr(C E_a C E_b)
        b_a = tr(C E_a)
        """
        # assign data
        if x1.ndim == 1:
            x1 = x1.reshape(-1, 1)
        self.x1 = x1
        self.Ndim = x1.ndim - 1
        # if x2 is not provided, just use x1
        if x2 is None:
            self.x2 = x1
        else:
            if x2.ndim == 1:
                x2 = x2.reshape(-1, 1)
            self.x2 = x2

        # assign metadata
        assert len(dx) == self.Ndim, "x1 must be of shape Ndim+1, dx of shape Ndim"
        self.dx = dx
        self.idx = idx
        self.scalar = scalar if scalar is not None else 1

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
            R = [r[idx] for r, idx in zip(R, self.idx)]

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

        # compute qft for each data dimension
        self.qft = [np.fft.fftshift(np.fft.ifft(np.eye(n[0])), axes=0) for n in shape]
        self.qft_pad = [np.pad(q, (n[1]-n[0])//2) for q, n in zip(self.qft, shape)]        

    def _compute_qR(self, qft, R):
        """
        Compute the product qft @ R

        Parameters
        ----------
        qft : list
            A list of length self.Ndim holding
            2D matrices of shape (Nbandpower, Npix)
            for each data dimension in x1.
        R : list
            A list of length self.Ndim holding
            the 2D weighting matrix for each
            data dimension in x1

        Returns
        -------
        list
        """
        return [q @ r for q, r in zip(qft, R)]

    def compute_qR(self):
        if not hasattr(self, 'qft'):
            raise NameError("Must first run self.compute_qft()")
        if not hasattr(self, 'R'):
            raise NameError("Must first run self.set_R()")
        self.qR = self._compute_qR(self.qft, self.R)

    def _compute_q(self, x1, qR, x2=None):
        """
        Compute the un-normalized bandpower
        q = 0.5 x1^T R^T qft qft^T R x2

        Parameters
        ----------
        x1 : ndarray
            LHS of QE, of shape (Npix1, Npix2, ..., Nother)
        qR : list of ndarray
            List of shape Ndim where Ndim is number of Npix
            dimensions in x1 (excluding Nother), holding
            FT and weighting matrices.
        x2 : ndarray, optional
            RHS of QE, default is to use x1.

        Returns
        -------
        ndarray of shape (Nbandpower1, Nbandpower2, ..., Nother)
        """
        es_str = 'abcdefgh'[:self.Ndim]
        for i, qr in enumerate(qR):
            str_in = es_str.replace(es_str[i], 'j')
            str_out = es_str.replace(es_str[i], 'i')
            x1 = np.einsum("ij,{}...->{}...".format(str_in, str_out), qr, x1)
            if x2 is not None:
                x2 = np.einsum("ij,{}...->{}...".format(str_in, str_out), qr, x2)

        if x2 is None:
            x2 = x1

        return 0.5 * x1.conj() * x2

    def compute_q(self):
        """
        Compute q: un-normalized band powers
        
        Results
        -------
        self.q
        """
        if not hasattr(self, 'qR'):
            raise NameError("Must first run self.compute_qR()")
        self.q = self._compute_q(self.x1, self.qR, self.x2)

    def _compute_H(self, qR, qft_pad):
        """
        Computes H matrix, mapping the true power spectrum
        to un-normalized power spectrum.

        q_a = H_ab p_b

        Note that H need not be square, e.g. to oversample
        the window functions in k space.

        Note that we compute a single H matrix for each
        data dimension in x1, i.e. for each element in self.qR

        Each matrix in qR is shape AXM, while matrix in qft_pad is
        shape BXM
        """
        H = []
        for qr, qp in zip(qR, qft_pad):
            H.append(0.5 * abs(qr @ qp.conj().T)**2)

        return H

    def compute_H(self):
        """
        Compute H_ab = 0.5 tr[R^T Q_a R Q_b]
        For R = C^-1, H = F is the Fisher matrix

        Parameters
        ----------
        enforce_real : bool
            If True, take real component of H matrix,
            assuming imaginary component is numerical noise.

        Results
        -------
        self.H
        """
        if not hasattr(self, 'qft_pad'):
            raise ValueError("First run compute_qft()")
        if not hasattr(self, 'qR'):
            raise ValueError("First run compute_qR()")

        self.H = self._compute_H(self.qR, self.qft_pad)

    def _compute_M(self, norm, H, rcond=1e-15, Wnorm=False):
        if norm == 'I':
            M = np.zeros(H.shape)
            M[range(len(H)), np.argmax(H, axis=1)] = 1 / H.sum(axis=1)
            M = M.T
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
                M = u @ np.diag(1/s) @ u.T.conj()
            elif norm == 'H^-1/2':
                M = u @ np.diag(1/np.sqrt(s)) @ u.T.conj() * np.sqrt(s).sum() / s.sum()
        else:
            raise ValueError("{} not recognized".format(norm))

        if Wnorm:
            # get window functions
            W = M @ H

            # ensure they are normalized
            M /= W.sum(axis=1, keepdims=True).clip(1e-3)

        return M * self.scalar

    def _compute_W(self, M, H):
        return M @ H

    def compute_MW(self, norm='I', rcond=1e-15, Wnorm=False):
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
        self.M, self.W
        """
        self.norm = norm
        # get normalization matrix
        assert hasattr(self, 'H'), "Must first run compute_H"
        self.M = [self._compute_M(norm, H, rcond=rcond, Wnorm=Wnorm) for H in self.H]

        # compute window functions
        self.W = [self._compute_W(M, H) for M, H in zip(self.M, self.H)]

    def _compute_p(self, M, q):
        """
        Compute the normalized bandpower
        p = M q

        Parameters
        ----------
        M : list of ndarray
            List holding 2D matrix of shape (Nbp, Nbp)
            for each bandpower dimension in self.q
        q : ndarray
            Un-normalized bandpower array of shape
            (Nbandpower1, Nbandpower2, ..., Nother)

        Returns
        -------
        ndarray of shape (Nbandpower1, Nbandpower2, ..., Nother)
        """
        es_str = 'abcdefgh'[:self.Ndim]
        p = q
        for i, m in enumerate(M):
            str_in = es_str.replace(es_str[i], 'j')
            str_out = es_str.replace(es_str[i], 'i')
            p = np.einsum("ij,{}...->{}...".format(str_in, str_out), m, p)

        return p

    def _compute_b(self, C, M, qR, rcond=1e-15):
        """
        Compute bias of normalized power spectrum

        Parameters
        ----------
        C : list of ndarray
            List holding a 2D covariance matrix of shape
            (Npix, Npix) for each data dimension in x1.
        M : list of ndarray
            List holding 2D matrix of shape (Nbp, Nbp)
            for each data dimension in x1.
        qR : list of ndarray
            List holding 2D qft @ R matrix of shape (Nbp, Npix)
            for each data dimension in x1.
        rcond : float
            Relative condition when decomposing C
            via svd

        Returns
        -------
        ndarray of shape self.q
        """
        # iterate over data dimensions
        b = np.zeros(self.x1.shape, dtype=float)
        if C is not None:
            for i, (c, m, qr) in enumerate(zip(C, M, qR)):
                if c is not None:
                    # decompose matrix c into A A^T
                    u, s, v = np.linalg.svd(c, hermitian=True)
                    keep = s > s.max() * rcond
                    A = u[:, keep] * np.diag(np.sqrt(s[keep]))

                    # compute qr A
                    qrA = qr @ A

                    # take abs sum to get un-normalized bias
                    ub = 0.5 * (abs(qrA)**2).sum(-1)

                    # normalize
                    nb = m @ ub

                    # sum bias terms
                    bshape = [1 for j in range(b.ndim)]
                    bshape[i] = -1
                    b += nb.reshape(bshape)

        return b

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
        self.p = self._compute_p(self.M, self.q)

        # compute bias term
        self.b = self._compute_b(C_bias, self.M, self.qR, rcond=rcond)

    def _compute_V(self, C, M, qR, diag=True):
        """
        Compute bandpower covariance

        2 tr[C E_a C E_b]
        
        where E_a = 1/2 M_ab R^T Q_b R

        Parameters
        ----------
        C : list of ndarray
            List holding a 2D covariance matrix of shape
            (Npix, Npix) for each data dimension in x1.
        M : list of ndarray
            List holding 2D matrix of shape (Nbp, Nbp)
            for each data dimension in x1.
        qR : list of ndarray
            List holding 2D qft @ R matrix of shape (Nbp, Npix)
            for each data dimension in x1.
        diag : bool, optional
            If True, only compute diagonal of
            bandpower covariance. Otherwise compute
            off-diagonal as well.
        Returns
        -------
        ndarray
        """
        V = []
        if C is None: C = [None for m in M]
        for i, (c, m, qr) in enumerate(zip(C, M, qR)):
            # default is identity
            v = np.eye(len(qr))
            # update if cov is provided
            if c is not None:
                # compute un-normalized E: qR outer product
                ue = 0.5 * np.einsum("ij,ik->ijk", qR, qR.conj())

                # compute normalized E
                e = np.einsum("ij,jkl->ikl", m, ue)

                # dot into covariance
                ec = np.einsum("ij,jkl->ikl", c, e)

                if diag:
                    # compute just variance
                    v = np.diag(np.einsum("ijk,ijk->i", ec, ec))

                else:
                    # compute full covariance
                    v = np.einsum("ijk,ljk->il", ec, ec)

            V.append(v)

        return V

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
        # compute bandpower covariance
        self.V = self._compute_V(C, self.M, self.qR, diag=diag)
        self.Vdiag = diag

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
                if self.Vdiag:
                    V = np.diag(np.kron(Vi[0].diagonal(), Vi[1].diagonal()))
                else:
                    V = np.kron(Vi[0], Vi[1])
                W = np.kron(Wi[0], Wi[1])
                K = np.meshgrid(ki[0], ki[1])
                k = np.sqrt(K[0].ravel()**2 + K[1].ravel()**2)
            else:
                if self.Vdiag:
                    V = np.diag(np.kron(np.kron(Vi[0].diagonal(), Vi[1].diagonal()), Vi[2].diagonal()))
                else:
                    V = np.kron(np.kron(Vi[0], Vi[1]), Vi[2])
                W = np.kron(Wi[0], np.kron(Wi[1], Wi[2]))
                K = np.meshgrid(ki[0], ki[1], ki[2])
                k = np.sqrt(K[0].ravel()**2 + K[1].ravel()**2 + K[2].ravel()**2)

            # TODO: V = kron(V1, V2) is not correct, V should preserve units of p^2
            # maybe its a geometric mean?

            # get k_avg points
            if k_avg is None:
                k_avg = np.linspace(k.min(), k.max(), max(len(ki[0]), len(ki[1]))//2)

        else:
            # otherwise, we are folding this axis
            assert axis < pi.ndim
            p = np.moveaxis(pi, axis, 0)
            b = np.moveaxis(bi, axis, 0)
            V = Vi[axis]
            W = Wi[axis]
            k = abs(ki[axis])

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
        if self.Vdiag:
            Vinv = np.diag(1/V.diagonal().clip(1e-30))
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

    def compute_MWVp(self, norm='I', rcond=1e-15, C_bias=None, C_errs=None, diag=True):
        """
        Shallow wrapper for compute_MW, p, V and spherical average

        Parameters
        ----------
        norm : see compute_MW()
        rcond : see compute_MW()
        C_bias : see compute_p()
        C_errs : see compute_V()
        diag : see compute_V()
        """
        self.compute_MW(norm=norm, rcond=rcond)
        self.compute_p(C_bias=C_bias)
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
    def __init__(self, x1, dx, kperp, x2=None, idx=None, scalar=None):
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
        Notes
        -----
        The code adopts the following defintions

        qft_a = e^{-2pi i a n / N}
        qR_a = qft_a R
        H_ab = 1/2 tr[qR_a^T qR_b]
        q_a = 1/2 x1^T qR_a^T qR_a x2
        M = H^-1 or H^-1/2 or propto I
        p_a = M_ab q_b
        W = M H
        E_a = 1/2 M_ab (qR_b^T qR_b)
        V_ab = 2 tr(C E_a C E_b)
        b_a = tr(C E_a)
        """
        super().__init__(x1, [1, dx], x2=x2, idx=[slice(None), idx], scalar=scalar)
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
        assert hasattr(self, 'R'), "Must first run set_R()"
        # compute k-modes for each data dimension
        shape = [R.shape for R in self.R]
        self.k = [kperp, np.fft.fftshift(2*np.pi*np.fft.fftfreq(self.x1.shape[1], self.dx[1]))]

        # compute qft for each data dimension
        self.qft = [np.eye(self.x1.shape[0]),
                    np.fft.fftshift(np.fft.ifft(np.eye(self.x1.shape[1])), axes=0)]
        self.qft_pad = [np.eye(self.x1.shape[0]),
                        np.pad(self.qft[1], abs(np.diff(self.R[1].shape)))]

