import numpy as np
import pandas as pd
from scipy.optimize import least_squares, newton, brenth
from scipy.integrate import quad
from scipy.stats import norm


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
                np.array([0.0, 0.0, 0.0, 0.0, -0.9]),
                np.array([100.0, 1.0, 1.0, 5.0, 0.9])
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

    def calibrate(self, data, px, r, cov_s):
        fit_res = []
        for i, a in enumerate(self.assets):
            model = self.univariates[a]
            res = model.calibrate(data[data.corp == a], px[i], r[i])
            fit_res.append(res)
            if hasattr(fit_res, 'x'):
                self.kappa_vec[i], self.v0_vec[i], self.theta_vec[i], \
                    self.eta_vec[i], self.rho_sv_vec[i] = fit_res.x
                self.r_vec[i] = r[i]
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


class Option(object):
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
        new_option = Option()
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
        new_option = Option()
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
        new_option = Option()
        new_option.payoff = lambda S: \
            self.payoff(S)*scalar
        new_option.pricing_func = lambda S: \
            self.price(S)*scalar
        new_option.delta_finite_diff = lambda S, dS: \
            self.delta_finite_diff(S, dS)*scalar
        new_option.gamma_finite_diff = lambda S, dS: \
            self.gamma_finite_diff(S, dS)*scalar
        return new_option


class Call(Option):
    def __init__(self, model, K, T):
        super(Call, self).__init__()
        self.model = model
        self.K, self.T = K, T
        self.pricing_func = model.make_pricing_function()
        self.payoff = lambda S: (S-self.K).clip(0)

    def price(self, S):
        return self.pricing_func(S, self.K, self.T)


class Put(Option):
    def __init__(self, model, K, T):
        super(Put, self).__init__()
        self.model = model
        self.K, self.T = K, T
        call_px = model.make_pricing_function()
        self.pricing_func = lambda S_, K_, T_: call_px(
            S_, K_, T_) + K_*np.exp(-model.r*T_) - S_
        self.payoff = lambda S: (self.K-S).clip(0)

    def price(self, S):
        return self.pricing_func(S, self.K, self.T)
    
    
class ZeroCostCollar(Option):
    def __init__(self, model, S, put_K, T):
        super(ZeroCostCollar, self).__init__()
        self.model = model
        self.S, self.put_K, self.T = S, put_K, T

        self.put = Put(model, put_K, T)
        self.put_px = self.put.price(S)
        call_px_func = model.make_pricing_function()
        self.call_K = newton(
            lambda K: call_px_func(S, K, T)-self.put_px, x0=put_K)
        self.call = Call(model, self.call_K, T)

        synthetic = (self.put - self.call)
        self.payoff = synthetic.payoff
        self.pricing_func = synthetic.pricing_func
        self.delta_finite_diff = synthetic.delta_finite_diff
        self.gamma_finite_diff = synthetic.gamma_finite_diff


