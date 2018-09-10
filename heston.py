import time

import numpy as np
import numba as nb
import pandas as pd
from scipy.optimize import least_squares, newton, brenth
from scipy.integrate import quad
from scipy.stats import norm

from progressbar import ProgressBar


def str3f_vector(x):
    __str = '['
    for i, _ in enumerate(x):
        __str += f'{x[i]:.3f}, '
    return __str[:-2]+']'


def bs_call_px(S, K, T, r, sigma, q=0):
    if np.isnan(sigma):
        return np.nan
    d1 = (np.log(S/K) + (r-q+.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    N1, N2 = norm.cdf(d1), norm.cdf(d2)
    return S*np.exp(-q*T)*N1 - K*np.exp(-r*T)*N2


def bs_vega(S, K, T, r, sigma, q=0):
    if np.isnan(sigma):
        return np.nan
    d1 = (np.log(S/K) + (r-q+.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return S * np.exp(-q*T) * np.sqrt(T) * norm.pdf(d1)


def bs_iv(c, S, K, T, r, q=0):
    f = lambda sigma: bs_call_px(S, K, T, r, sigma, q) - c
    df = lambda sigma: bs_vega(S, K, T, r, sigma, q)
    sol = newton(f, 0.5, df, tol=1e-6)
    if np.abs(sol) > 2:
        return np.nan
    return sol


def make_surface(func, x, y):
    xx, yy = np.meshgrid(x, y)
    func_v = np.vectorize(func)
    zz = func_v(xx, yy)
    return xx, yy, zz


class UnivariateHestonSV(object):

    def __init__(self, asset_identifier):
        self.asset_idx = asset_identifier
        self.r = None
        self.kappa = None
        self.v0 = None
        self.theta = None
        self.eta = None
        self.rho_sv = None
        self.calibrated = False

    def set_params(self, params_list):
        for i, k in enumerate(
                ['r', 'kappa', 'v0', 'theta', 'eta', 'rho_sv']):
            setattr(self, k, params_list[i])
        self.calibrated = True

    @staticmethod
    def heston_call_px(S0, K, T, r, kappa, v0, theta, eta, rho_sv):
        def _phi(w, t):
            gamma = eta ** 2 / 2
            beta = kappa - rho_sv * eta * w * 1j
            alpha = -(w ** 2 / 2) - (1j * w / 2)
            h = np.sqrt(beta ** 2 - 4 * alpha * gamma)
            r_plus = (beta + h) / (eta ** 2)
            r_minus = (beta - h) / (eta ** 2)
            g = r_minus / r_plus
            eht = np.exp(-h * t)
            D = r_minus * ((1 - eht) / (1 - g * eht))
            C = kappa * (r_minus * t - (2 / (eta ** 2)) * np.log(
                (1 - g * eht) / (1 - g)))
            return np.exp(
                C * theta + D * v0 + 1j * w * np.log(S0 * np.exp(r * t)))

        def _integrand_1(w):
            f = (np.exp(-1j * w * np.log(K)) * _phi(w - 1j, T)) / (
                1j * w * _phi(-1j, T))
            return f.real

        def _integrand_2(w):
            f = (np.exp(-1j * w * np.log(K)) * _phi(w, T)) / (1j * w)
            return f.real

        p1 = 0.5 + (1 / np.pi) * quad(_integrand_1, 0, 100)[0]
        p2 = 0.5 + (1 / np.pi) * quad(_integrand_2, 0, 100)[0]
        return S0 * p1 - np.exp(-r * T) * K * p2

    def calibrate(self, ivs_data, S0, r):
        ivs_data = ivs_data.reset_index()

        def _err_func(params):
            kappa, v0, theta, eta, rho_sv = params
            sq_errs = np.zeros(len(ivs_data))
            for i, row in ivs_data.iterrows():
                if np.isnan(row.iv):
                    sq_errs[i] = 0
                    continue
                mkt_px = row.bs_px
                heston_px = self.heston_call_px(
                    S0, row.strike, row.mat, r,
                    kappa, v0, theta, eta, rho_sv
                )
                sq_errs[i] = mkt_px - heston_px
            return sq_errs

        fit_res = least_squares(
            _err_func, np.array([1.0, 0.01, 0.35, 0.7, -0.4]),
            bounds=(
                np.array([0.0, 0.0, 0.0, 0.0, -0.8]),
                np.array([100.0, 0.2, 1.0, 5.0, 0.8])
            ),
            ftol=1e-3, verbose=2,
        )
        if hasattr(fit_res, 'x'):
            self.kappa, self.v0, self.theta, \
                self.eta, self.rho_sv = fit_res.x
            self.r = r
        self.calibrated = True
        return fit_res

    def make_iv_surface_function(self, S0):
        assert self.calibrated

        def _f(K, T):
            heston_px = self.heston_call_px(
                S0, K, T, self.r,
                self.kappa, self.v0, self.theta, self.eta, self.rho_sv
            )
            return bs_iv(heston_px, S0, K, T, self.r)

        return _f

    def make_pricing_function(self):
        assert self.calibrated

        def _p(S, K, T):
            return self.heston_call_px(
                S, K, T, self.r,
                self.kappa, self.v0, self.theta, self.eta, self.rho_sv
            )

        return _p


class MultivariateHestonSV(object):

    def __init__(self, assets):
        self.n_assets = len(assets)
        self.assets = assets
        self.univariates = {a: UnivariateHestonSV(a) for a in assets}

        self.r_vec = np.zeros((1, self.n_assets))
        self.kappa_vec = np.zeros((1, self.n_assets))
        self.v0_vec = np.zeros((1, self.n_assets))
        self.theta_vec = np.zeros((1, self.n_assets))
        self.eta_vec = np.zeros((1, self.n_assets))
        self.rho_sv_vec = np.zeros((1, self.n_assets))

        self.cov = None
        self.calibrated = False

    def set_params(self, params_dict):
        for i, a in enumerate(self.assets):
            params_list_a = [params_dict['r']]+list(params_dict[a])
            self.univariates[a].set_params(params_list_a)
            self.r_vec[0, i], self.kappa_vec[0, i], self.v0_vec[0, i], \
                self.theta_vec[0, i], self.eta_vec[0, i], \
                self.rho_sv_vec[0, i] = params_list_a
        rho_sv_flat = self.rho_sv_vec.reshape(self.n_assets, )
        self.cov = np.block([
            [np.eye(self.n_assets), np.diag(rho_sv_flat)],
            [np.diag(rho_sv_flat), params_dict['cov_s']]
        ])
        self.calibrated = True

    def calibrate(self, data, px, r, cov_s):
        fit_res = []
        for i, a in enumerate(self.assets):
            model = self.univariates[a]
            res = model.calibrate(data[data.corp == a], px[i], r[i])
            fit_res.append(res)
            if hasattr(res, 'x'):
                self.kappa_vec[0, i], self.v0_vec[0, i], \
                    self.theta_vec[0, i], self.eta_vec[0, i], \
                    self.rho_sv_vec[0, i] = res.x
                self.r_vec[0, i] = r[i]
        rho_sv_flat = self.rho_sv_vec.reshape(self.n_assets,)
        self.cov = np.block([
            [np.eye(self.n_assets), np.diag(rho_sv_flat)],
            [np.diag(rho_sv_flat), cov_s]
        ])
        self.calibrated = True
        return fit_res

    def make_iv_surface_functions(self, px):
        assert self.calibrated
        funcs = dict()
        for i, a in enumerate(self.assets):
            funcs[a] = self.univariates[a].make_iv_surface_function(px[i])
        return funcs

    def make_pricing_functions(self):
        assert self.calibrated
        funcs = dict()
        for i, a in enumerate(self.assets):
            funcs[a] = self.univariates[a].make_pricing_function()
        return funcs

    @staticmethod
    @nb.jit(nb.types.UniTuple(nb.float64[:, :], 3)(
        nb.int64, nb.int64, nb.float64, nb.float64[:],
        nb.float64[:, :], nb.float64[:, :], nb.float64[:, :],
        nb.float64[:, :], nb.float64[:, :], nb.float64[:, :]))
    def simulate_1path(n_nodes, n_assets, T, S0,
                       r_vec, kappa_vec, v0_vec, theta_vec, eta_vec, cov):
        dt = T / n_nodes
        # simulate drivers
        dW = np.random.multivariate_normal(np.array(
            [0] * (2 * n_assets)), cov, n_nodes)
        dW_v = dW[:, :n_assets]
        dW_X = dW[:, n_assets:]

        # simulate heston process
        V = np.zeros((n_nodes, n_assets))
        X = np.zeros((n_nodes, n_assets))  # X_t = log (S_t)
        V[0, :], X[0, :] = v0_vec, np.log(S0)

        for t in range(1, n_nodes):
            # V[t,:] -> (1*n_assets) row vector
            # V_{t+1} = V_t + k(theta - V_t^+)dt + eta*\sqrt{dt V_t^+} dW_v
            V[t, :] = V[t - 1, :] + kappa_vec * (
                theta_vec - V[t - 1, :].clip(0)) * dt + eta_vec * (
                np.sqrt(V[t - 1, :].clip(0) * dt)) * dW_v[t - 1, :]

            # S_{t+1} = S_t + (r-.5 V_t^+)dt + \sqrt{dt V_t^+} dW_S
            X[t, :] = X[t - 1, :] + \
                (r_vec - .5 * V[t - 1, :].clip(0)) * dt + \
                (np.sqrt(V[t - 1, :].clip(0) * dt)) * dW_X[t - 1, :]
        S = np.exp(X)
        return S, X, V

    @staticmethod
    @nb.jit(nb.types.UniTuple(nb.float64[:, :], 7)(
        nb.int64, nb.int64, nb.float64, nb.float64, nb.float64[:],
        nb.float64[:, :], nb.float64[:, :], nb.float64[:, :],
        nb.float64[:, :], nb.float64[:, :], nb.float64[:, :]))
    def simulate_1path_cfb(n_nodes, n_assets, T, dS, S0,
                           r_vec, kappa_vec, v0_vec,
                           theta_vec, eta_vec, cov):
        dt = T / n_nodes
        # simulate drivers
        dW = np.random.multivariate_normal(np.array(
            [0] * (2 * n_assets)), cov, n_nodes)
        dW_v = dW[:, :n_assets]
        dW_X = dW[:, n_assets:]

        # containers
        V = np.zeros((n_nodes, n_assets))
        X_central = np.zeros((n_nodes, n_assets))  # X_t = log (S_t)
        X_fwd = np.zeros((n_nodes, n_assets, n_assets))
        # X_fwd[t, i, j] = log (S_it | S_j0 + dS_j0)
        X_bwd = np.zeros((n_nodes, n_assets, n_assets))

        V[0, :], X_central[0, :] = v0_vec, np.log(S0)
        S0_matrix = np.repeat(S0.reshape(n_assets, 1), n_assets, axis=1)
        X_fwd[0, :, :] = np.log(S0_matrix + np.diag(np.repeat(dS, n_assets)))
        X_bwd[0, :, :] = np.log(S0_matrix - np.diag(np.repeat(dS, n_assets)))

        # simulate heston process
        for t in range(1, n_nodes):
            # V[t,:] -> (1*n_assets) row vector
            # V_{t+1} = V_t + k(theta - V_t^+)dt + eta*\sqrt{dt V_t^+} dW_v
            V[t, :] = V[t - 1, :] + kappa_vec * (
                theta_vec - V[t - 1, :].clip(0)) * dt + eta_vec * (
                np.sqrt(V[t - 1, :].clip(0) * dt)) * dW_v[t - 1, :]
            # S_{t+1} = S_t + (r-.5 V_t^+)dt + \sqrt{dt V_t^+} dW_S
            X_central[t, :] = X_central[t - 1, :] + \
                (r_vec - .5 * V[t - 1, :].clip(0)) * dt + \
                (np.sqrt(V[t - 1, :].clip(0) * dt)) * dW_X[t - 1, :]
            for X in [X_bwd, X_fwd]:
                for j in range(n_assets):
                    # new process with respect to (S_0j +- dS_0j)
                    X[t, :, j] = X[t - 1, :, j] + \
                        (r_vec - .5 * V[t - 1, :].clip(0)) * dt + \
                        (np.sqrt(V[t - 1, :].clip(0) * dt)) * dW_X[t - 1, :]

        S_central = np.exp(X_central)
        S_fwd = np.exp(X_fwd)
        S_bwd = np.exp(X_bwd)
        return S_central, S_fwd, S_bwd, X_central, X_fwd, X_bwd,  V

    @nb.jit
    def simulate_paths(self, n_paths, n_nodes, T, S0):
        SS = np.zeros((n_nodes, n_paths, self.n_assets))
        XX = np.zeros((n_nodes, n_paths, self.n_assets))
        VV = np.zeros((n_nodes, n_paths, self.n_assets))
        bar = ProgressBar()
        for i in bar(list(range(n_paths))):
            S, X, V = MultivariateHestonSV.simulate_1path(
                n_nodes, self.n_assets, T, S0,
                self.r_vec, self.kappa_vec, self.v0_vec, self.theta_vec,
                self.eta_vec, self.cov)
            SS[:, i, :], XX[:, i, :], VV[:, i, :] = S, X, V
        return SS, XX, VV

    @nb.jit
    def simulate_paths_cfb(self, n_paths, n_nodes, T, dS, S0):
        SSc = np.zeros((n_nodes, n_paths, self.n_assets))
        SSf = np.zeros((n_nodes, n_paths, self.n_assets, self.n_assets))
        SSb = np.zeros((n_nodes, n_paths, self.n_assets, self.n_assets))
        XXc = np.zeros((n_nodes, n_paths, self.n_assets))
        XXf = np.zeros((n_nodes, n_paths, self.n_assets, self.n_assets))
        XXb = np.zeros((n_nodes, n_paths, self.n_assets, self.n_assets))
        VV = np.zeros((n_nodes, n_paths, self.n_assets))
        bar = ProgressBar()
        for i in bar(list(range(n_paths))):
            Sc, Sf, Sb, Xc, Xf, Xb, V = \
                MultivariateHestonSV.simulate_1path_cfb(
                    n_nodes, self.n_assets, T, dS, S0,
                    self.r_vec, self.kappa_vec, self.v0_vec, self.theta_vec,
                    self.eta_vec, self.cov)
            SSc[:, i, :], XXc[:, i, :], VV[:, i, :] = Sc, Xc, V
            SSf[:, i, :, :], XXf[:, i, :, :] = Sf, Xf
            SSb[:, i, :, :], XXb[:, i, :, :] = Sb, Xb
        return SSc, SSf, SSb, XXc, XXf, XXb, VV


class UnivariateOption(object):
    def __init__(self):
        self.pricing_func = lambda S: 0
        self.payoff = lambda S: 0
        self.delta_finite_diff = lambda S, dS: \
            (self.price(S+dS/2) - self.price(S-dS/2)) / dS
        self.gamma_finite_diff = lambda S, dS: \
            (self.price(S+dS) + self.price(S-dS)
             - 2*self.price(S)) / (dS**2)

    def price(self, S):
        return self.pricing_func(S)

    def delta(self, S, dS=1e-4):
        return self.delta_finite_diff(S, dS)

    def gamma(self, S, dS=1e-4):
        return self.gamma_finite_diff(S, dS)

    def __add__(self, other):
        new_option = UnivariateOption()
        new_option.payoff = lambda S: \
            self.payoff(S) + other.payoff(S)
        new_option.pricing_func = lambda S: \
            self.price(S) + other.price(S)
        new_option.delta_finite_diff = lambda S, dS: \
            self.delta_finite_diff(S, dS) + other.delta_finite_diff(S, dS)
        new_option.gamma_finite_diff = lambda S, dS: \
            self.gamma_finite_diff(S, dS) + other.gamma_finite_diff(S, dS)
        return new_option

    def __sub__(self, other):
        new_option = UnivariateOption()
        new_option.payoff = lambda S: \
            self.payoff(S) - other.payoff(S)
        new_option.pricing_func = lambda S: \
            self.price(S) - other.price(S)
        new_option.delta_finite_diff = lambda S, dS: \
            self.delta_finite_diff(S, dS) - other.delta_finite_diff(S, dS)
        new_option.gamma_finite_diff = lambda S, dS: \
            self.gamma_finite_diff(S, dS) - other.gamma_finite_diff(S, dS)
        return new_option

    def __mul__(self, scalar):
        new_option = UnivariateOption()
        new_option.payoff = lambda S: \
            self.payoff(S)*scalar
        new_option.pricing_func = lambda S: \
            self.price(S)*scalar
        new_option.delta_finite_diff = lambda S, dS: \
            self.delta_finite_diff(S, dS)*scalar
        new_option.gamma_finite_diff = lambda S, dS: \
            self.gamma_finite_diff(S, dS)*scalar
        return new_option


class MultiAssetsOption(object):
    def __init__(self, multivariate_model, Ks, T):
        self.model = multivariate_model
        self.assets = self.model.assets
        self.n_assets = len(self.assets)
        self.Ks, self.T = Ks, T

    def payoff(self, paths):
        raise NotImplementedError

    def mc_price(self, spots, n_paths=1000, n_nodes_per_year=32,
                 return_paths=False, pre_computed_paths=None):
        time.sleep(0.5)
        if hasattr(self, 'force_monitor_freq'):
            n_nodes_per_year = self.force_monitor_freq
        if hasattr(self, 'force_disc_factor_dim'):
            disc_factor_dim = self.force_disc_factor_dim
        else:
            disc_factor_dim = self.n_assets
        if pre_computed_paths:
            paths = pre_computed_paths
        else:
            print('Simulating paths...')
            paths = self.model.simulate_paths(
                n_paths, n_nodes_per_year*self.T, self.T, spots)
        sample = self.payoff(paths)
        if disc_factor_dim > 1:
            disc_factor = np.exp(
                -self.model.r_vec*self.T).reshape(disc_factor_dim,)
        else:
            disc_factor = np.exp(-self.model.r_vec[0, 0]*self.T)
        px = np.mean(sample, axis=0)*disc_factor
        se = np.std(sample, axis=0) / np.sqrt(n_paths)
        if return_paths:
            return px, se, paths
        return px, se

    @staticmethod
    def mc_price_custom_payoff(paths, disc_factor, payoff):
        sample = payoff(paths)
        px = np.mean(sample, axis=0)*disc_factor
        se = np.std(sample, axis=0) / np.sqrt(len(paths))
        return px, se

    def mc_gamma(self, spots, dS=1e-1,
                 n_paths=1000, n_nodes_per_year=32):
        print('Simulating paths...')
        SSc, SSf, SSb, XXc, XXf, XXb, VV = self.model.simulate_paths_cfb(
            n_paths, max(int(n_nodes_per_year * self.T), 32),
            self.T, dS, spots
        )
        paths_central = (SSc, XXc, VV)
        paths_fwd = (SSf, XXf, VV)
        paths_bwd = (SSb, XXb, VV)
        sample_central = self.payoff(paths_central)
        sample_fwd = self.payoff(paths_fwd)
        sample_bwd = self.payoff(paths_bwd)
        disc_factor = np.exp(-self.model.r_vec * self.T).reshape(
            self.n_assets, )
        px_central = np.mean(sample_central, axis=0)*disc_factor
        px_fwd = np.mean(sample_fwd, axis=0)*disc_factor
        px_bwd = np.mean(sample_bwd, axis=0)*disc_factor
        gamma = (px_fwd+px_bwd-2*px_central) / (dS**2)
        # se = np.std(sample, axis=0) / np.sqrt(n_paths)
        return gamma

    def mc_delta(self, spots, dS=1e-4,
                 n_paths=1000, n_nodes_per_year=32,
                 pre_computed_paths=None):
        if hasattr(self, 'force_disc_factor_dim'):
            disc_factor_dim = self.force_disc_factor_dim
        else:
            disc_factor_dim = self.n_assets
        if pre_computed_paths:
            SSc, SSf, SSb, XXc, XXf, XXb, VV = pre_computed_paths
        else:
            print('Simulating paths...')
            SSc, SSf, SSb, XXc, XXf, XXb, VV = self.model.simulate_paths_cfb(
                n_paths, max(int(n_nodes_per_year * self.T), 32),
                self.T, dS, spots
            )
        if disc_factor_dim > 1:
            disc_factor = np.exp(
                -self.model.r_vec*self.T).reshape(disc_factor_dim,)
        else:
            disc_factor = np.exp(-self.model.r_vec[0, 0]*self.T)
        deltas = []
        for j in range(self.n_assets):
            paths_fwd = (SSf[:, :, :, j], XXf[:, :, :, j], VV)
            paths_bwd = (SSb[:, :, :, j], XXb[:, :, :, j], VV)
            sample_fwd = self.payoff(paths_fwd)
            sample_bwd = self.payoff(paths_bwd)
            px_fwd = np.mean(sample_fwd, axis=0)*disc_factor
            px_bwd = np.mean(sample_bwd, axis=0)*disc_factor
            delta = np.array((px_fwd-px_bwd) / (2*dS))
            delta = delta.reshape(disc_factor_dim, 1)
            deltas.append(delta)
        # se = np.std(sample, axis=0) / np.sqrt(n_paths)
        return np.hstack(deltas)


class MultiAssetsAsianCall(MultiAssetsOption):
    def __init__(self, multivariate_model, Ks, T):
        super(MultiAssetsAsianCall, self).__init__(
            multivariate_model, Ks, T)

    def __repr__(self):
        return f'MultiAssetsAsianCall: K={str3f_vector(self.Ks)}, T={self.T}'

    def payoff(self, paths):
        SS, XX, VV = paths
        S_avg = np.mean(SS, axis=0)
        return (S_avg - np.array(self.Ks)).clip(0)


class MultiAssetsAsianPut(MultiAssetsOption):
    def __init__(self, multivariate_model, Ks, T):
        super(MultiAssetsAsianPut, self).__init__(
            multivariate_model, Ks, T)

    def __repr__(self):
        return f'MultiAssetsAsianPut: K={str3f_vector(self.Ks)}, T={self.T}'

    def payoff(self, paths):
        SS, XX, VV = paths
        S_avg = np.mean(SS, axis=0)
        return (np.array(self.Ks) - S_avg).clip(0)


class MultiAssetsDiscreteKIEuropeanPut(MultiAssetsOption):
    def __init__(self, multivariate_model, Ks, Hs, T, monitor_freq=252):
        super(MultiAssetsDiscreteKIEuropeanPut, self).__init__(
            multivariate_model, Ks, T)
        self.Hs = Hs
        self.force_monitor_freq = monitor_freq

    def __repr__(self):
        return f'MultiAssetsDiscreteKIAsianPut: K={str3f_vector(self.Ks)}, ' \
               + f'H={str3f_vector(self.Hs)}, T={self.T}'

    def payoff(self, paths):
        SS, XX, VV = paths
        S_min = np.min(SS, axis=0)
        return (self.Ks - SS[-1, :, :]).clip(0) * (S_min < self.Hs)


class MultiAssetsWorstOfDiscreteKIEuropeanPut(MultiAssetsOption):
    def __init__(self, multivariate_model, collars, best_of_call, spots,
                 T_entry, T_mature, Ds, Ks=None, Hs=None, verbose=2,
                 monitor_freq=252, n_paths=5000, pre_computed_paths=None,
                 premium=0.0):
        super(MultiAssetsWorstOfDiscreteKIEuropeanPut, self).__init__(
            multivariate_model, None, T_mature)
        self.premium = premium
        self.Ds = Ds
        self.Ks = Ks
        self.Hs = Hs
        self.S_init = spots
        self.T_mature = T_mature
        self.T_entry = T_entry
        self.collars = collars
        self.best_of_call = best_of_call
        self.force_monitor_freq = monitor_freq
        self.force_disc_factor_dim = 1

        if Ks is not None:
            return
        if verbose:
            print('Solving barriers...')
        self.boc_px, _, paths = self.best_of_call.mc_price(
            spots, n_paths, monitor_freq, return_paths=True,
            pre_computed_paths=pre_computed_paths)
        disc_factor = np.exp(-self.model.r_vec[0, 0] * self.T)

        def _err_func(_Ks):
            # make a different payoff function for every K
            def _woKIPut_payoff(_paths):

                SS, XX, VV = _paths
                n_t, n, n_a = SS.shape
                _idx_entry = int(n_t / self.T_mature) * self.T_entry - 1
                _S_entry = SS[_idx_entry, :, :]
                _ret = (_S_entry - self.S_init) / self.S_init
                _worst_idx = np.argmin(_ret, axis=1)
                _S_min = np.min(
                    SS[_idx_entry:, :, :], axis=0)[range(n), _worst_idx]
                __Ks = np.vstack([_Ks] * n)[
                    range(n), _worst_idx]
                __Hs = np.vstack([_Ks-self.Ds] * n)[
                    range(n), _worst_idx]
                _x = np.vstack([self.collars.shares] * n)[
                    range(n), _worst_idx]
                _S_terminal = SS[-1, range(n), _worst_idx]
                return (__Ks - _S_terminal).clip(0) * (_S_min < __Hs) * _x

            # use same paths
            KI_px, _ = MultiAssetsOption.mc_price_custom_payoff(
                paths, disc_factor, _woKIPut_payoff)
            return (self.boc_px - KI_px) + premium

        fit_res = least_squares(
            _err_func, x0=self.collars.put_Ks,
            bounds=(
                np.zeros(self.n_assets),
                1.5*self.collars.put_Ks
            ),
            ftol=1e-5, verbose=verbose,
        )
        if hasattr(fit_res, 'x'):
            self.Ks = fit_res.x
            self.Hs = self.Ks - self.Ds
        self.err_func = _err_func

    def __repr__(self):
        return f'MultiAssetsWorstOfDiscreteKIEuropeanPut: ' \
               + f'K={str3f_vector(self.Ks)}, ' \
               + f'H={str3f_vector(self.Hs)}, ' \
               + f'T_entry={self.T_entry}, T_mature={self.T_mature}'

    def payoff(self, paths):
        SS, XX, VV = paths
        n_t, n_paths, n_a = SS.shape
        idx_entry = int(n_t / self.T_mature) * self.T_entry - 1
        S_entry = SS[idx_entry, :, :]
        ret = (S_entry - self.S_init) / self.S_init
        worst_idx = np.argmin(ret, axis=1)
        S_min = np.min(
            SS[idx_entry:, :, :], axis=0)[range(n_paths), worst_idx]
        Ks = np.vstack([self.Ks] * n_paths)[
            range(n_paths), worst_idx]
        Hs = np.vstack([self.Hs] * n_paths)[
            range(n_paths), worst_idx]
        x = np.vstack([self.collars.shares] * n_paths)[
            range(n_paths), worst_idx]
        S_terminal = SS[-1, range(n_paths), worst_idx]
        return (Ks - S_terminal).clip(0) * (S_min < Hs) * x


class MultiAssetsBestOfAsianCall(MultiAssetsOption):
    def __init__(self, multivariate_model, collars,
                 T_entry, T_mature, S_init, required_return):
        super(MultiAssetsBestOfAsianCall, self).__init__(
            multivariate_model, None, T_mature)
        self.T_mature = T_mature
        self.T_entry = T_entry
        self.S_init = S_init
        self.collars = collars
        self.required_ret = required_return
        self.force_disc_factor_dim = 1

    def __repr__(self):
        return f'MultiAssetsBestOfAsianCall: ' \
               + f'K={str3f_vector(self.collars.call_Ks)}, ' \
               + f'R_required={self.required_ret}, ' \
               + f'T_entry={self.T_entry}, T_mature={self.T_mature}'

    def payoff(self, paths):
        SS, XX, VV = paths
        n_t, n_paths, n_a = SS.shape
        idx_entry = int(n_t / self.T_mature) * self.T_entry - 1
        S_entry = SS[idx_entry, :, :]
        ret = (S_entry - self.S_init) / self.S_init
        best_idx = np.argmax(ret, axis=1)
        best_ret = np.max(ret, axis=1)
        K = np.vstack([self.collars.call_Ks]*n_paths)[
            range(n_paths), best_idx]
        x = np.vstack([self.collars.shares] * n_paths)[
            range(n_paths), best_idx]
        S_avg = np.mean(SS, axis=0)[range(n_paths), best_idx]
        return (S_avg - K).clip(0) * (best_ret >= self.required_ret) * x


class MultiAssetsBestOfEuropeanCall(MultiAssetsOption):
    def __init__(self, multivariate_model, collars,
                 T_entry, T_mature, S_init, required_return):
        super(MultiAssetsBestOfEuropeanCall, self).__init__(
            multivariate_model, None, T_mature)
        self.T_mature = T_mature
        self.T_entry = T_entry
        self.S_init = S_init
        self.collars = collars
        self.required_ret = required_return
        self.force_disc_factor_dim = 1

    def __repr__(self):
        return f'MultiAssetsBestOfEuropeanCall: ' \
               + f'K={str3f_vector(self.collars.call_Ks)}, ' \
               + f'R_required={self.required_ret}, ' \
               + f'T_entry={self.T_entry}, T_mature={self.T_mature}'

    def payoff(self, paths):
        SS, XX, VV = paths
        n_t, n_paths, n_a = SS.shape
        idx_entry = int(n_t / self.T_mature) * self.T_entry - 1
        S_entry = SS[idx_entry, :, :]
        ret = (S_entry - self.S_init) / self.S_init
        best_idx = np.argmax(ret, axis=1)
        best_ret = np.max(ret, axis=1)
        K = np.vstack([self.collars.call_Ks]*n_paths)[
            range(n_paths), best_idx]
        x = np.vstack([self.collars.shares]*n_paths)[
            range(n_paths), best_idx]
        _S_terminal = SS[-1, range(n_paths), best_idx]
        return (_S_terminal - K).clip(0) * (best_ret >= self.required_ret) * x


class MultiAssetsAsianZeroCostCollar(MultiAssetsOption):
    def __init__(self, multivariate_model, spots, T, put_Ks, x=None,
                 call_Ks=None, n_paths=5000, n_nodes_per_year=32,
                 verbose=2, pre_computed_paths=None, premium=None):
        super(MultiAssetsAsianZeroCostCollar, self).__init__(
            multivariate_model, None, T)
        if premium is not None:
            self.premium = premium
        else:
            self.premium = np.zeros(self.n_assets)
        self.spots, self.put_Ks, self.T = spots, put_Ks, T
        if x is not None:
            self.shares = x
        else:
            self.shares = np.ones(self.n_assets)

        self.put_leg = MultiAssetsAsianPut(multivariate_model, put_Ks, T)
        self.call_leg = MultiAssetsAsianCall(multivariate_model, call_Ks, T)
        if call_Ks is not None:
            self.call_Ks = call_Ks
            return
        # otherwise solve call_Ks
        if verbose:
            print('Solving call strikes...')
        put_px, _, paths = self.put_leg.mc_price(
            spots, n_paths, n_nodes_per_year,
            return_paths=True, pre_computed_paths=pre_computed_paths)
        disc_factor = np.exp(-self.model.r_vec * self.T).reshape(
            self.n_assets, )

        def _err_func(_call_Ks):
            # make a different payoff function for every K
            def _call_payoff(_paths):
                S_avg = np.mean(_paths[0], axis=0)
                return (S_avg-np.array(_call_Ks)).clip(0)
            # use same paths
            call_px, _ = self.call_leg.mc_price_custom_payoff(
                paths, disc_factor, _call_payoff)
            return (put_px - call_px) + self.premium

        fit_res = least_squares(
            _err_func, x0=spots,
            bounds=(
                np.array([0.001]*self.n_assets),
                np.array([np.inf]*self.n_assets)
            ),
            ftol=1e-5, verbose=verbose,
        )
        if hasattr(fit_res, 'x'):
            self.call_Ks = fit_res.x
        self.call_leg.Ks = self.call_Ks

    def __repr__(self):
        return f'Zero Cost Collar:\n+ ' + \
               repr(self.put_leg)+'\n- '+repr(self.call_leg)

    def payoff(self, paths):
        SS, XX, VV = paths
        S_avg = np.mean(SS, axis=0)
        return (np.array(self.put_Ks) - S_avg).clip(0) - (
            S_avg - np.array(self.call_Ks)).clip(0)


class EuropeanCall(UnivariateOption):
    def __init__(self, model, K, T):
        super(EuropeanCall, self).__init__()
        self.model = model
        self.K, self.T = K, T
        self.pricing_func = model.make_pricing_function()
        self.payoff = lambda S: (S-self.K).clip(0)

    def price(self, S):
        return self.pricing_func(S, self.K, self.T)


class EuropeanPut(UnivariateOption):
    def __init__(self, model, K, T):
        super(EuropeanPut, self).__init__()
        self.model = model
        self.K, self.T = K, T
        call_px = model.make_pricing_function()
        self.pricing_func = lambda S_, K_, T_: call_px(
            S_, K_, T_) + K_*np.exp(-model.r*T_) - S_
        self.payoff = lambda S: (self.K-S).clip(0)

    def price(self, S):
        return self.pricing_func(S, self.K, self.T)
    
    
class EuropeanZeroCostCollar(UnivariateOption):
    def __init__(self, model, S, put_K, T):
        super(EuropeanZeroCostCollar, self).__init__()
        self.model = model
        self.S, self.put_K, self.T = S, put_K, T

        self.put = EuropeanPut(model, put_K, T)
        self.put_px = self.put.price(S)
        call_px_func = model.make_pricing_function()
        self.call_K = newton(
            lambda K: call_px_func(S, K, T)-self.put_px, x0=put_K)
        self.call = EuropeanCall(model, self.call_K, T)

        synthetic = (self.put - self.call)
        self.payoff = synthetic.payoff
        self.pricing_func = synthetic.pricing_func
        self.delta_finite_diff = synthetic.delta_finite_diff
        self.gamma_finite_diff = synthetic.gamma_finite_diff


def slippage_sq_root(vol, x, adv, s_0):
    pct_px_slippage = 0.15 * vol * np.sqrt(
        (np.abs(x) / adv - 0.3).clip(0))
    px_slippage = pct_px_slippage * s_0
    return px_slippage


def structure_constructor(model, x, put_Ks, Ds, R_req,
                          adv, adv_thres, S0, cap_required, alpha,
                          pre_computed_paths, premiums, dS=1e-4):
    SSc, SSf, SSb, XXc, XXf, XXb, VV = pre_computed_paths
    paths_c = (SSc, XXc, VV)
    # paths_f = (SSf, XXf, VV)
    # paths_b = (SSb, XXb, VV)
    n_nodes, n_paths, n_assets = SSc.shape
    disc_factor = np.exp(-0.045 * 2)
    vol_0 = np.sqrt(model.v0_vec)
    premium_collar, premium_KI = premiums
    # construct portfolio
    zero_collar = MultiAssetsAsianZeroCostCollar(
        model, spots=S0, x=np.abs(x),
        put_Ks=put_Ks, T=2,
        n_paths=n_paths, n_nodes_per_year=252,
        premium=premium_collar,
        pre_computed_paths=paths_c, verbose=0
    )
    best_of_call = MultiAssetsBestOfAsianCall(
        model, zero_collar,
        T_entry=1, T_mature=2,
        S_init=S0, required_return=R_req
    )
    worst_of_KI_put = MultiAssetsWorstOfDiscreteKIEuropeanPut(
        model, zero_collar, best_of_call, S0, Ds=Ds,
        T_entry=1, T_mature=2, n_paths=n_paths,
        premium=premium_KI,
        pre_computed_paths=paths_c, verbose=0
    )
    structure = {
        'Zero Collar': zero_collar,
        'Best of Call': best_of_call,
        'worst of KI Put': worst_of_KI_put
    }
    # calculate initial delta
    delta_collar = zero_collar.mc_delta(
        spots=S0, dS=dS, pre_computed_paths=pre_computed_paths)
    delta_boc = best_of_call.mc_delta(
        spots=S0, dS=dS, pre_computed_paths=pre_computed_paths)
    delta_KI = worst_of_KI_put.mc_delta(
        spots=S0, dS=dS, pre_computed_paths=pre_computed_paths)
    delta_collar = np.diag(delta_collar).reshape(1, n_assets)*x
    delta_total = (delta_collar - delta_boc + delta_KI)
    pct_adv = delta_total / adv
    pct_adv_above = (pct_adv - adv_thres).clip(0)
    px_slippage = slippage_sq_root(vol_0, delta_total, adv, S0)
    cap_locked_in = np.sum(-put_Ks * x) * disc_factor
    loss = np.linalg.norm(pct_adv_above, 2) + alpha * (
        cap_required-cap_locked_in).clip(0)
    deltas = [delta_collar, delta_boc, delta_KI]
    return loss, structure, pct_adv, cap_locked_in, deltas, px_slippage


def structure_constructor_best_eu(model, x, put_Ks, Ds, R_req,
                                  adv, adv_thres, S0, cap_required, alpha,
                                  pre_computed_paths, premiums, dS=1e-4):
    SSc, SSf, SSb, XXc, XXf, XXb, VV = pre_computed_paths
    paths_c = (SSc, XXc, VV)
    # paths_f = (SSf, XXf, VV)
    # paths_b = (SSb, XXb, VV)
    n_nodes, n_paths, n_assets = SSc.shape
    disc_factor = np.exp(-0.045 * 2)
    vol_0 = np.sqrt(model.v0_vec)
    premium_collar, premium_KI = premiums
    # construct portfolio
    zero_collar = MultiAssetsAsianZeroCostCollar(
        model, spots=S0, x=np.abs(x),
        put_Ks=put_Ks, T=2,
        n_paths=n_paths, n_nodes_per_year=252,
        premium=premium_collar,
        pre_computed_paths=paths_c, verbose=0
    )
    best_of_call = MultiAssetsBestOfEuropeanCall(
        model, zero_collar,
        T_entry=1, T_mature=2,
        S_init=S0, required_return=R_req
    )
    worst_of_KI_put = MultiAssetsWorstOfDiscreteKIEuropeanPut(
        model, zero_collar, best_of_call, S0, Ds=Ds,
        T_entry=1, T_mature=2, n_paths=n_paths,
        premium=premium_KI,
        pre_computed_paths=paths_c, verbose=0
    )
    structure = {
        'Zero Collar': zero_collar,
        'Best of Call': best_of_call,
        'worst of KI Put': worst_of_KI_put
    }
    # calculate initial delta
    delta_collar = zero_collar.mc_delta(
        spots=S0, dS=dS, pre_computed_paths=pre_computed_paths)
    delta_boc = best_of_call.mc_delta(
        spots=S0, dS=dS, pre_computed_paths=pre_computed_paths)
    delta_KI = worst_of_KI_put.mc_delta(
        spots=S0, dS=dS, pre_computed_paths=pre_computed_paths)
    delta_collar = np.diag(delta_collar).reshape(1, n_assets)*x
    delta_total = (delta_collar - delta_boc + delta_KI)
    pct_adv = delta_total / adv
    pct_adv_above = (pct_adv - adv_thres).clip(0)
    px_slippage = slippage_sq_root(vol_0, delta_total, adv, S0)
    cap_locked_in = np.sum(-put_Ks * x) * disc_factor
    loss = np.linalg.norm(pct_adv_above, 2) + alpha * (
        cap_required-cap_locked_in).clip(0)
    deltas = [delta_collar, delta_boc, delta_KI]
    return loss, structure, pct_adv, cap_locked_in, deltas, px_slippage
