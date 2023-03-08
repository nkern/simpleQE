"""
Test suite for simpleqe.utils
"""

import numpy as np
from scipy import signal

from simpleqe import qe, utils


def prep_data(freqs, data_spw=None, pspec_spw=None, seed=None, ind_noise=True, Ntimes=200):
    # assume freqs to be in Hz
    Nfreqs = len(freqs)
    Ntimes = 200
    Cf = lambda x: utils.gauss_cov(x, 10.0, var=2e2)
    Ce = lambda x: utils.exp_cov(x, 1.0, var=1e-3)
    Cn = lambda x: utils.diag_cov(x, var=1e-2)
    return utils.gen_data(freqs, Cf, Ce, Cn, Ntimes=Ntimes, data_spw=data_spw, pspec_spw=pspec_spw)


def test_cosmo():
    # test init
    cosmo = utils.Cosmology()
    # test defaults parameters
    assert np.isclose(cosmo.H0.value, 67.7)
    assert np.isclose(cosmo.Om0, 0.3075)
    assert np.isclose(cosmo.Ob0, 0.0486)
    assert np.isclose(cosmo.Odm0, 0.2589)
    assert np.isclose(cosmo.Ode0, 0.6910070182)
    # test basic calculations
    assert np.isclose(cosmo.H(10), 1375.866236)
    assert np.isclose(cosmo.f2z(100e6), 13.20405751)
    assert np.isclose(cosmo.z2f(cosmo.f2z(100e6)), 100e6)


def test_interp_Wcdf():
    # setup data: simple blackman-harris test
    freqs = np.linspace(140e6, 160e6, 100, endpoint=False)
    D, F, E, N = prep_data(freqs, Ntimes=200)
    t = np.diag(signal.windows.blackmanharris(len(freqs)))
    D.set_R(t), D.compute_qft(); D.compute_H(); D.compute_q()
    D.compute_MWVp(norm='I', C_bias=F.C, C_errs=D.C)
    D.average_bandpowers(); D.compute_dsq()

    # compute window function bounds
    med, low, hi = utils.interp_Wcdf(D.W[1], D.k[1])
    # assert med is close to kp
    assert np.isclose(med, D.k[1], atol=np.diff(D.k[1])[0]).all()
    # assert symmetric low / hi (except for boundaries)
    assert np.isclose(low[3:-3], hi[3:-3], atol=1e-10).all()


def test_ravel_mats():
    # test identity broadcasting
    f = np.linspace(100, 120, 100)
    Cg = utils.gauss_cov(f, 1) * (f/110)**-2.2
    out = utils.ravel_mats(10, Cg)
    assert len(out) == 1000
    assert np.isclose(out.diagonal()[:100], Cg.diagonal()).all()


def test_cov():
    freqs = np.linspace(140e6, 160e6, 100, endpoint=False)
    Nfreqs = len(freqs)
    gauss = utils.gauss_cov(freqs/1e6, 5)
    expon = utils.exp_cov(freqs/1e6, 5)
    diag = utils.diag_cov(freqs/1e6)
    assert gauss.shape == (Nfreqs, Nfreqs)
    assert expon.shape == (Nfreqs, Nfreqs)
    assert diag.shape == (Nfreqs, Nfreqs)

