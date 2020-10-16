"""
Test suite for simpleqe.qe
"""

import numpy as np
import h5py
import ast
from scipy import signal

from simpleqe import qe, utils
from simpleqe.data import DATA_PATH


def prep_data(freqs, data_spw=None, pspec_spw=None, seed=None, ind_noise=True, Ntimes=200):
    # assume freqs to be in Hz
    Nfreqs = len(freqs)
    Ntimes = 200
    Cf = lambda x: utils.gauss_cov(x, 10.0, var=2e2)
    Ce = lambda x: utils.exp_cov(x, 1.0, var=1e-3)
    Cn = lambda x: utils.diag_cov(x, var=1e-2)
    return utils.gen_data(freqs, Cf, Ce, Cn, Ntimes=Ntimes, data_spw=data_spw, pspec_spw=pspec_spw)


def test_QE_object():
    # test basic QE
    # setup data
    freqs = np.linspace(140e6, 160e6, 100, endpoint=False)
    D, F, E, N = prep_data(freqs, Ntimes=200)
    I = np.eye(D.Nfreqs)
    D.set_R(I), D.compute_Q(); D.compute_H(); D.compute_q()
    D.compute_MWVp(norm='I', C_bias=F.C, C_data=D.C); D.compute_dsq()
    for p in ['p', 'V', 'b', 'W', 'p_sph', 'V_sph', 'b_sph', 'W_sph', 'dsq', 'dsq_b', 'dsq_V']:
        assert hasattr(D, p)


def normalization_test():
    # test normalization against hera_pspec
    # get data
    with h5py.File(DATA_PATH + "/zen.2458116.30448.HH.C.uvh5") as f:
        freqs = f['Header']['freq_array'][:].squeeze()
        data = f['Data']['visdata'][:].squeeze()
    # get pre-computed power spectra from hera_pspec [dated 10/2020]
    with h5py.File(DATA_PATH + "/zen.2458116.30448.HH.C.h5") as f:
        pspec = f['data_spw0'][:].squeeze()
        window_func = f['window_function_spw0'][0].squeeze()
        c = ast.literal_eval(f.attrs['cosmo'])
        cosmo = utils.Cosmology(H0=c['H0'], Om0=c['Om_M'], Ob0=c['Om_b'])
        OP, OPP, bf = f['OmegaP'][:].squeeze(), f['OmegaPP'][:].squeeze(), f.attrs['beam_freqs']
        Oeff = np.interp(freqs, bf, OP**2 / OPP).mean()

    # run simpleQE
    D = qe.QE(freqs, data, cosmo=cosmo, Omega_Eff=Oeff)
    t = np.diag(signal.windows.blackmanharris(len(freqs)))
    D.set_R(t), D.compute_Q(); D.compute_H(); D.compute_q()
    D.compute_MWVp(norm='I')

    # compare normalization to hera_pspec
    ratio = D.p.T[:, ::-1] / pspec  # inverse fft convention wrt hera_pspec
    assert np.abs(1 - ratio.real).max() < 0.01  # assert the same to 1%

    # compare window function
    window_func[window_func.real < 1e-5] = 1  # only compare non-near-zero values
    D.W[D.W.real < 1e-5] = 1
    assert np.isclose(window_func.real / D.W.real, 1, atol=0.01).all()  # compare to 1%
