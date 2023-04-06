'''
BSD 3-Clause License

Copyright (c) 2021, Shota Imaki
All rights reserved.

https://github.com/simaki/adaptive-elastic-net
'''

import numbers
import warnings

import cvxpy
import numpy as np
from asgl import ASGL
from sklearn.base import MultiOutputMixin
from sklearn.base import RegressorMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet
from sklearn.linear_model._coordinate_descent import _alpha_grid
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted
from sklearn.linear_model._coordinate_descent import LinearModelCV

class AdaptiveElasticNet(ASGL, ElasticNet, MultiOutputMixin, RegressorMixin):
    """
    Objective function is
        (1 / 2 n_samples) * sum_i ||y_i - y_pred_i||^2
            + alpha * l1ratio * sum_j |coef_j|
            + alpha * (1 - l1ratio) * sum_j w_j * ||coef_j||^2
        w_j = |b_j| ** (-gamma)
        b_j = coefs obtained by fitting ordinary elastic net
        i: sample
        j: feature
        |X|: abs
        ||X||: square norm
    Parameters
    ----------
    - alpha : float, default=1.0
        Constant that multiplies the penalty terms.
    - l1_ratio : float, default=0.5
        float between 0 and 1 passed to ElasticNet
        (scaling between l1 and l2 penalties).
    - gamma : float > 0, default=1.0
        To guarantee the oracle property, following inequality should be satisfied:
            gamma > 2 * nu / (1 - nu)
            nu = lim(n_samples -> inf) [log(n_features) / log(n_samples)]
        default is 1 because this value is natural in the sense that
        l1_penalty / l2_penalty is not (directly) dependent on scale of features
    - fit_intercept = True
    - max_iter : int, default 10000
        The maximum number of iterations.
    - positive : bool, default=False
        When set to `True`, forces the coefficients to be positive.
    - positive_tol : float, optional
        Numerical optimization (cvxpy) may return slightly negative coefs.
        (See cvxpy issue/#1201)
        If coef > -positive_tol, ignore this and forcively set negative coef to zero.
        Otherwise, raise ValueError.
        If `positive_tol=None` always ignore (default)
    - eps_coef : float, default 1e-6
        Small constant to prevent zero division in b_j ** (-gamma).
    Attributes
    ----------
    - coef_
    - intercept_
    - enet_coef_
    - weights_
    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_features=2, random_state=0)
    >>> model = AdaptiveElasticNet(random_state=42)
    >>> model.fit(X, y)
    AdaptiveElasticNet(random_state=42, solver='default', tol=1e-05)
    >>> print(model.coef_)
    [14.24414426 48.9550584 ]
    >>> print(model.enet_coef_)
    [18.8... 64.5...]
    >>> print(model.weights_)
    [0.0530... 0.0154...]
    >>> print(model.intercept_)
    2.092...
    >>> print(model.predict([[0, 0]]))
    [2.092...]
    Constraint:
    >>> X, y = make_regression(n_features=10, random_state=0)
    >>> model = AdaptiveElasticNet(positive=True).fit(X, -y)
    >>> model.coef_
    array([ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        , 13.39621927,  0.        ])
    """

    def __init__(
        self,
        alpha=1.0,
        *,
        l1_ratio=0.5,
        gamma=1.0,
        fit_intercept=True,
        precompute=False,
        max_iter=10000,
        copy_X=True,
        solver=None,
        tol=None,
        positive=False,
        positive_tol=1e-3,
        random_state=None,
        eps_coef=1e-6,
    ):
        params_asgl = dict(model="lm", penalization="asgl")
        if solver is not None:
            params_asgl["solver"] = solver
        if tol is not None:
            params_asgl["tol"] = tol

        super().__init__(**params_asgl)

        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.gamma = gamma
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.precompute = precompute
        self.copy_X = copy_X
        self.positive = positive
        self.positive_tol = positive_tol
        self.random_state = random_state
        self.eps_coef = eps_coef

        if not self.fit_intercept:
            raise NotImplementedError

        # TODO(simaki) guarantee reproducibility.  is cvxpy reproducible?

    def fit(self, X, y, check_input=True):
        if self.alpha == 0:
            warnings.warn(
                "With alpha=0, this algorithm does not converge "
                "well. You are advised to use the LinearRegression "
                "estimator",
                stacklevel=2,
            )

        if not isinstance(self.l1_ratio, numbers.Number) or not 0 <= self.l1_ratio <= 1:
            raise ValueError(
                "l1_ratio must be between 0 and 1; " f"got l1_ratio={self.l1_ratio}"
            )

        if check_input:
            X_copied = self.copy_X and self.fit_intercept
            X, y = self._validate_data(
                X,
                y,
                accept_sparse="csc",
                order="F",
                dtype=[np.float64, np.float32],
                copy=X_copied,
                multi_output=True,
                y_numeric=True,
            )

        self.coef_, self.intercept_, self.enet_coef_, self.weights_ = self._ae(X, y)

        self.dual_gap_ = np.array([np.nan])
        self.n_iter_ = 1

        return self

    def predict(self, X):
        check_is_fitted(self, ["coef_", "intercept_"])
        return super(ElasticNet, self).predict(X)

    def elastic_net(self, **params):
        """
        ElasticNet with the same parameters of self.
        Parameters
        ----------
        - **params
            Overwrite parameters.
        Returns
        -------
        elastic_net : ElasticNet
        """
        elastic_net = ElasticNet()

        for k, v in self.get_params().items():
            try:
                elastic_net = elastic_net.set_params(**{k: v})
            except ValueError:
                # ElasticNet does not expect parameter `gamma`
                pass

        elastic_net = elastic_net.set_params(**params)

        return elastic_net

    def _ae(self, X, y) -> (np.array, float):
        """
        Adaptive elastic-net counterpart of ASGL.asgl
        Returns
        -------
        (coef, intercept, enet_coef, weights)
            - coef : np.array, shape (n_features,)
            - intercept : float
            - enet_coef : np.array, shape (n_features,)
            - weights : np.array, shape (n_features,)
        """
        check_X_y(X, y)

        n_samples, n_features = X.shape
        beta_variables = [cvxpy.Variable(n_features)]
        # _, beta_variables = self._num_beta_var_from_group_index(group_index)
        # beta_variables = cvxpy.Variable()

        model_prediction = 0.0

        if self.fit_intercept:
            beta_variables = [cvxpy.Variable(1)] + beta_variables
            ones = cvxpy.Constant(np.ones((n_samples, 1)))
            model_prediction += ones @ beta_variables[0]

        # --- define objective function ---
        #   l1 weights w_i are identified with coefs in usual elastic net
        #   l2 weights nu_i are fixed to unity in adaptive elastic net

        # /2 * n_samples to make it consistent with sklearn (asgl uses /n_samples)
        model_prediction += X @ beta_variables[1]
        error = cvxpy.sum_squares(y - model_prediction) / (2 * n_samples)

        enet_coef = self.elastic_net().fit(X, y).coef_
        weights = 1.0 / (np.maximum(np.abs(enet_coef), self.eps_coef) ** self.gamma)

        # XXX: we, paper by Zou Zhang and sklearn use norm squared for l2_penalty
        # whereas asgl uses norm itself
        l1_coefs = self.alpha * self.l1_ratio * weights
        l2_coefs = self.alpha * (1 - self.l1_ratio) * 1.0
        l1_penalty = cvxpy.Constant(l1_coefs) @ cvxpy.abs(beta_variables[1])
        l2_penalty = cvxpy.Constant(l2_coefs) * cvxpy.sum_squares(beta_variables[1])

        constraints = [b >= 0 for b in beta_variables] if self.positive else []

        # --- optimization ---
        problem = cvxpy.Problem(
            cvxpy.Minimize(error + l1_penalty + l2_penalty), constraints=constraints
        )
        # OSQP seems to be default for our problem.
        problem.solve(solver="OSQP", max_iter=self.max_iter)

        if problem.status != "optimal":
            raise ConvergenceWarning(
                f"Solver did not reach optimum (Status: {problem.status})"
            )

        beta_sol = np.concatenate([b.value for b in beta_variables], axis=0)
        beta_sol[np.abs(beta_sol) < self.tol] = 0

        intercept, coef = beta_sol[0], beta_sol[1:]

        # Check if constraint violation is less than positive_tol. cf cvxpy issue/#1201
        if self.positive and self.positive_tol is not None:
            if not all(c.value(self.positive_tol) for c in constraints):
                raise ValueError(f"positive_tol is violated. coef is:\n{coef}")
        coef = np.maximum(coef, 0)

        self.solver_stats = problem.solver_stats

        return (coef, intercept, enet_coef, weights)

    # def _weights_from_elasticnet(self, X, y) -> np.array:
    #     """
    #     Determine weighs by fitting ElasticNet

    #     wj of (2.1) in Zou-Zhang 2009

    #     Returns
    #     -------
    #     weights : np.array, shape (n_features,)
    #     """
    #     abscoef = np.maximum(np.abs(ElasticNet().fit(X, y).coef_), self.eps_coef)
    #     weights = 1 / (abscoef ** self.gamma)

    #     return weights

    @classmethod
    def aenet_path(
        cls,
        X,
        y,
        *,
        l1_ratio=0.5,
        eps=1e-3,
        n_alphas=100,
        alphas=None,
        precompute="auto",
        Xy=None,
        copy_X=True,
        coef_init=None,
        verbose=False,
        return_n_iter=False,
        positive=False,
        check_input=True,
        **params,
    ):
        """
        Return regression results for multiple alphas
        see enet_path in sklearn
        Returns
        -------
        (alphas, coefs, dual_gaps)
            XXX dual_gaps are nan
        """

        if alphas is None:
            alphas = _alpha_grid(
                X,
                y,
                Xy=Xy,
                l1_ratio=l1_ratio,
                fit_intercept=False,
                eps=eps,
                n_alphas=n_alphas,
                copy_X=False,
            )

        n_samples, n_features = X.shape

        dual_gaps = np.empty(n_alphas)
        n_iters = []
        coefs = np.empty((n_features, n_alphas), dtype=X.dtype)
        coef_ = np.zeros(coefs.shape[:-1], dtype=X.dtype, order="F")
        for i, alpha in enumerate(alphas):
            model = cls(alpha=alpha)
            model.fit(X, y)

            coef_ = model.coef_

            coefs[..., i] = coef_
            dual_gaps[i] = np.nan
            n_iters.append(1)

        return (alphas, coefs, dual_gaps)

class AdaptiveElasticNetCV(RegressorMixin, LinearModelCV):
    """
    AdaptiveElasticNet with CV
    Parameters
    ----------
    - l1_ratio : float, default=0.5
        The ElasticNet mixing parameter, with 0 <= l1_ratio <= 1.
    - n_alphas :
    - alphas :
    - gamma : float, default 1.0
        To guarantee the oracle property, following inequality should be satisfied:
            gamma > 2 * nu / (1 - nu)
            nu = lim(n_samples -> inf) [log(n_features) / log(n_samples)]
        default is 1 because this value is natural in the sense that
        l1_penalty / l2_penalty is not (directly) dependent on scale of features
    - fit_intercept : bool, default True
        Whether to calculate the intercept for this model.
        For now False is not allowed
    - cv : int, cross-validation generator or iterable, default=None
        Determines the cross-validation splitting strategy.
    - eps : float, default=1e-3
        Length of the path.
        eps=1e-3 means that alpha_min / alpha_max = 1e-3.
    - positive : bool, default=False
        When set to True, forces the coefficients to be positive.
    - positive_tol : float, optional
        Numerical optimization (cvxpy) may return slightly negative coefs.
        (See cvxpy issue/#1201)
        If coef > -positive_tol, ignore this and forcively set negative coef to zero.
        Otherwise, raise ValueError.
        If `positive_tol=None` always ignore (default)
    TODO
    ----
    cv wrt gamma?
    Notes
    -----
    (simaki)
        accoding to https://projecteuclid.org/download/pdfview_1/euclid.aos/1245332831
        condition gamma > 2 nu / 1 - nu is necessary to guarantee oracle property
        nu = log(n_features) / log(n_samples), which is in range (0, 1)
        hmm:
        Also note that, in the finite dimension setting, ν = 0; thus, any positive gamma
        can be used, which agrees with the results in Zou (2006).
    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_features=2, random_state=0)
    >>> model = AdaptiveElasticNetCV(cv=5)
    >>> model.fit(X, y)
    AdaptiveElasticNetCV(cv=5)
    >>> print(model.alpha_)
    0.199...
    >>> print(model.intercept_)
    0.706...
    >>> print(model.predict([[0, 0]]))
    [0.706...]
    >>> X, y = make_regression(n_features=10, random_state=0)
    >>> model = AdaptiveElasticNetCV(positive=True).fit(X, -y)
    >>> model.coef_
    array([1.16980429e-04, 2.14503535e-05, 0.00000000e+00, 4.45525264e-05,
           3.00411576e-04, 1.26646882e-04, 0.00000000e+00, 1.42388065e-04,
           2.05464198e-03, 0.00000000e+00])
    """

    path = AdaptiveElasticNet.aenet_path

    def __init__(
        self,
        *,
        l1_ratio=0.5,
        n_alphas=100,
        alphas=None,
        gamma=1.0,
        fit_intercept=True,
        # precompute="auto",
        cv=None,
        # copy_X=True,
        # selection="cyclic",
        eps=1e-3,
        positive=False,
        positive_tol=None,
        # normalize=False,
        precompute="auto",
    ):
        super().__init__(
            n_alphas=n_alphas,
            alphas=alphas,
            fit_intercept=fit_intercept,
            # normalize=normalize,
            precompute=precompute,
            # precompute=precompute,
            cv=cv,
            # copy_X=copy_X,
            # selection=selection,
            eps=eps,
        )

        self.l1_ratio = l1_ratio
        self.gamma = gamma
        self.eps = eps
        self.positive = positive
        self.positive_tol = positive_tol

    def _get_estimator(self):
        return AdaptiveElasticNet(
            l1_ratio=self.l1_ratio,
            gamma=self.gamma,
            positive=self.positive,
            positive_tol=self.positive_tol,
            tol = 1e-3, # increase tol to avoid converge warning
        )

    def _is_multitask(self):
        return False
