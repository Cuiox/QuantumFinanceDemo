from math import exp, log, sqrt
from typing import List, Optional, Tuple, Union

import numpy as np
from qiskit import QuantumCircuit
from qiskit.algorithms import EstimationProblem, IterativeAmplitudeEstimation
from qiskit.circuit.library import LinearAmplitudeFunction
from qiskit_aer.primitives import Sampler
from scipy.stats import norm


class LogNormalDistribution(QuantumCircuit):
    def __init__(
        self,
        num_qubits: int,
        mu: float,
        sigma: float,
        bounds: Tuple[float, float],
        upto_diag: bool = False,
        name: str = "P(X)",
    ) -> None:
        """
        Args:
            num_qubits: The number of qubits used to discretize the random variable. For a 1d
                random variable, ``num_qubits`` is an integer, for multiple dimensions a list
                of integers indicating the number of qubits to use in each dimension.
            mu: The parameter :math:`\mu` of the distribution.
                Can be either a float for a 1d random variable or a list of floats for a higher
                dimensional random variable.
            sigma: The parameter :math:`\sigma^2` or :math:`\Sigma`, which is the variance or
                covariance matrix.
            bounds: The truncation bounds of the distribution as tuples. For multiple dimensions,
                ``bounds`` is a list of tuples ``[(low0, high0), (low1, high1), ...]``.
                If ``None``, the bounds are set to ``(0, 1)`` for each dimension.
            upto_diag: If True, load the square root of the probabilities up to multiplication
                with a diagonal for a more efficient circuit.
            name: The name of the circuit.
        """
        inner = QuantumCircuit(num_qubits, name=name)
        x = np.linspace(bounds[0], bounds[1], num=2**num_qubits)
        # compute the normalized, truncated probabilities
        probabilities = []
        from scipy.stats import multivariate_normal

        for x_i in x:
            # map probabilities from normal to log-normal reference:
            # https://stats.stackexchange.com/questions/214997/multivariate-log-normal-probabiltiy-density-function-pdf
            if np.min(x_i) > 0:
                det = 1 / np.prod(x_i)
                probability = multivariate_normal.pdf(np.log(x_i), mu, sigma) * det
            else:
                probability = 0
            probabilities += [probability]
        normalized_probabilities = probabilities / np.sum(probabilities)  # type: ignore

        # store as properties
        self._values = x
        self._probabilities = normalized_probabilities
        self._bounds = bounds

        super().__init__(*inner.qregs, name=name)

        # use default the isometry (or initialize w/o resets) algorithm to construct the circuit
        # pylint: disable=no-member
        if upto_diag:
            inner.isometry(np.sqrt(normalized_probabilities), inner.qubits, None)
            self.append(inner.to_instruction(), inner.qubits)  # Isometry is not a Gate
        else:
            from qiskit.extensions import (
                Initialize,
            )  # pylint: disable=cyclic-import

            initialize = Initialize(np.sqrt(normalized_probabilities))
            circuit = initialize.gates_to_uncompute().inverse()
            inner.compose(circuit, inplace=True)
            self.append(inner.to_gate(), inner.qubits)

    @property
    def values(self) -> np.ndarray:
        """Return the discretized points of the random variable."""
        return self._values

    @property
    def probabilities(self) -> np.ndarray:
        """Return the sampling probabilities for the values."""
        return self._probabilities  # type: ignore

    @property
    def bounds(self) -> Union[Tuple[float, float], List[Tuple[float, float]]]:
        """Return the bounds of the probability distribution."""
        return self._bounds


class BSM(object):
    """
    A class to implement the Black-Scholes-Merton (BSM) model for option pricing.

    Attributes:
        _stock_price (float): The current price of the stock.
        _strike_price (float): The strike price of the option.
        _expiration (float): The time to expiration of the option in years.
        _risk_free_rate (float): The risk-free interest rate.
        _volatility (float): The volatility of the stock.
    """

    def __init__(
        self,
        stock_price: float,
        strike_price: float,
        expiration: float,
        risk_free_rate: float,
        volatility: float,
    ) -> None:
        """
        Initialize the BSM model with the given parameters.

        Args:
            stock_price (float): The current price of the stock.
            strike_price (float): The strike price of the option.
            expiration (float): The time to expiration of the option in years.
            risk_free_rate (float): The risk-free interest rate.
            volatility (float): The volatility of the stock.
        """
        self._stock_price = stock_price
        self._strike_price = strike_price
        self._expiration = expiration
        self._risk_free_rate = risk_free_rate
        self._volatility = volatility

    def d1(self) -> float:
        """
        Calculate the d1 term in the BSM formula.

        Returns:
            float: The value of d1.
        """
        return (
            log(self._stock_price / self._strike_price)
            + (self._risk_free_rate + self._volatility**2 / 2.0) * self._expiration
        ) / (self._volatility * sqrt(self._expiration))

    def d2(self) -> float:
        """
        Calculate the d2 term in the BSM formula.

        Returns:
            float: The value of d2.
        """
        return self.d1() - self._volatility * sqrt(self._expiration)

    def bs_call(self) -> float:
        """
        Calculate the price of a European call option using the BSM formula.

        Returns:
            float: The price of the call option.
        """
        return self._stock_price * norm.cdf(self.d1()) - self._strike_price * exp(
            -self._risk_free_rate * self._expiration
        ) * norm.cdf(self.d2())

    def bs_put(self) -> float:
        """
        Calculate the price of a European put option using the BSM formula.

        Returns:
            float: The price of the put option.
        """
        return (
            norm.cdf(-self.d2())
            * self._strike_price
            * exp(-self._risk_free_rate * self._expiration)
            - norm.cdf(-self.d1()) * self._stock_price
        )

    def solve(self, mode: str = "call") -> float:
        """
        Solve for the option price based on the specified mode (call or put).

        Args:
            mode (str): The type of option to price. Can be "call" or "put".

        Returns:
            float: The calculated price of the option.

        Raises:
            ValueError: If the mode specified is not "call" or "put".
        """
        if mode == "call":
            return self.bs_call()
        elif mode == "put":
            return self.bs_put()
        else:
            raise ValueError(f"mode: {mode} must be either 'call' or 'put'.")


class CMC(object):
    """
    A class to implement the Classical Monte Carlo (CMC) method for option pricing.

    Attributes:
        _stock_price (float): The current price of the stock.
        _strike_price (float): The strike price of the option.
        _expiration (float): The time to expiration of the option in years.
        _risk_free_rate (float): The risk-free interest rate.
        _volatility (float): The volatility of the stock.
        simulation_count (int): The number of Monte Carlo simulations to run.
        trading_days_per_month (int): The number of trading days in a month.
        trading_days (int): The total number of trading days over the option's life.
    """

    def __init__(
        self,
        stock_price: float,
        strike_price: float,
        expiration: float,
        risk_free_rate: float,
        volatility: float,
        simulation_count: int = 5000,
        trading_days_per_month: int = 21,
    ) -> None:
        """
        Initialize the CMC model with the given parameters.

        Args:
            stock_price (float): The current price of the stock.
            strike_price (float): The strike price of the option.
            expiration (float): The time to expiration of the option in years.
            risk_free_rate (float): The risk-free interest rate.
            volatility (float): The volatility of the stock.
            simulation_count (int, optional): The number of Monte Carlo simulations to run. Default is 5000.
            trading_days_per_month (int, optional): The number of trading days in a month. Default is 21.
        """
        self._stock_price = stock_price
        self._strike_price = strike_price
        self._expiration = expiration
        self._risk_free_rate = risk_free_rate
        self._volatility = volatility
        self.simulation_count = simulation_count
        self.trading_days_per_month = trading_days_per_month
        self.trading_days = int(self._expiration * 12 * self.trading_days_per_month)

    def solve(self, mode: str = "call") -> float:
        """
        Solve for the option price using the CMC method based on the specified mode (call or put).

        Args:
            mode (str): The type of option to price. Can be "call" or "put".

        Returns:
            float: The calculated price of the option.

        Raises:
            ValueError: If the mode specified is not "call" or "put".
        """
        # np.random.seed(seed_value)
        t = self._expiration / self.trading_days
        drift = (self._risk_free_rate - self._volatility**2 / 2) * t
        epsilon = np.random.normal(0, 1, (self.simulation_count, self.trading_days))
        a = self._volatility * np.sqrt(t)

        prices = np.zeros((self.simulation_count, self.trading_days))
        prices[:, 0] = self._stock_price

        for timestep in range(1, self.trading_days):
            prices[:, timestep] = prices[:, timestep - 1] * np.exp(
                drift + a * epsilon[:, timestep]
            )

        if mode == "call":
            callPayoffs = np.maximum(prices[:, -1] - self._strike_price, 0)
            Value = np.mean(callPayoffs) * np.exp(
                -self._risk_free_rate * self._expiration
            )
        elif mode == "put":
            putPayoffs = np.maximum(self._strike_price - prices[:, -1], 0)
            Value = np.mean(putPayoffs) * np.exp(
                -self._risk_free_rate * self._expiration
            )
        else:
            raise ValueError(f"mode: {mode} must be either 'call' or 'put'.")

        return Value


class QMC(object):
    """
    A class to implement the Quantum Monte Carlo (QMC) method for option pricing.

    Attributes:
        _stock_price (float): The current price of the stock.
        _strike_price (float): The strike price of the option.
        _expiration (float): The time to expiration of the option in years.
        _risk_free_rate (float): The risk-free interest rate.
        _volatility (float): The volatility of the stock.
        num_uncertainty_qubits (int): Number of qubits to represent uncertainty.
        bound_param (float): Bound parameter for the log-normal distribution.
        c_approx (float): Approximation scaling factor for the payoff function.
        epsilon (float): Target precision for amplitude estimation.
        alpha (float): Confidence level for amplitude estimation.
        shots (int): Number of shots (simulations) for the quantum algorithm.
        ae_seed (Optional[int]): Seed for amplitude estimation randomness.
    """

    def __init__(
        self,
        stock_price: float,
        strike_price: float,
        expiration: float,
        risk_free_rate: float,
        volatility: float,
        num_uncertainty_qubits: int = 3,
        bound_param: float = 3.0,
        c_approx: float = 0.03,
        epsilon: float = 0.003,  # 0.0003,
        alpha: float = 0.05,
        shots: int = 100,
        ae_seed: Optional[int] = 42,
    ) -> None:
        """
        Initialize the QMC model with the given parameters.

        Args:
            stock_price (float): The current price of the stock.
            strike_price (float): The strike price of the option.
            expiration (float): The time to expiration of the option in years.
            risk_free_rate (float): The risk-free interest rate.
            volatility (float): The volatility of the stock.
            num_uncertainty_qubits (int, optional): Number of qubits to represent uncertainty. Default is 3.
            bound_param (float, optional): Bound parameter for the log-normal distribution. Default is 3.0.
            c_approx (float, optional): Approximation scaling factor for the payoff function. Default is 0.03.
            epsilon (float, optional): Target precision for amplitude estimation. Default is 0.003.
            alpha (float, optional): Confidence level for amplitude estimation. Default is 0.05.
            shots (int, optional): Number of shots for the quantum algorithm. Default is 100.
            ae_seed (Optional[int], optional): Seed for amplitude estimation randomness. Default is 42.
        """
        self._stock_price = stock_price
        self._strike_price = strike_price
        self._expiration = expiration
        self._risk_free_rate = risk_free_rate
        self._volatility = volatility
        self.num_uncertainty_qubits = num_uncertainty_qubits
        self.bound_param = bound_param
        self.c_approx = c_approx
        self.epsilon = epsilon
        self.alpha = alpha
        self.shots = shots
        self.ae_seed = ae_seed

    def get_mu_sigma(self) -> Tuple[float, float]:
        """
        Calculate the parameters mu and sigma for the log-normal distribution.

        Returns:
            Tuple[float, float]: The calculated mu and sigma values.
        """
        mu = (
            self._risk_free_rate - 0.5 * self._volatility**2
        ) * self._expiration + np.log(self._stock_price)
        sigma = self._volatility * np.sqrt(self._expiration)
        return mu, sigma

    def get_bounds(self) -> Tuple[float, float]:
        """
        Calculate the bounds for the log-normal distribution.

        Returns:
            Tuple[float, float]: The lower and upper bounds for the distribution.
        """
        # resulting parameters for log-normal distribution
        mu, sigma = self.get_mu_sigma()
        mean = np.exp(mu + sigma**2 / 2)
        variance = (np.exp(sigma**2) - 1) * np.exp(2 * mu + sigma**2)
        stddev = np.sqrt(variance)
        # lowest and highest value considered for the spot price; in between, an equidistant discretization is considered.
        low = np.maximum(0, mean - self.bound_param * stddev)
        high = mean + self.bound_param * stddev
        bounds = (low, high)
        return bounds

    def get_uncertainty_model(
        self, bounds: Tuple[float, float]
    ) -> LogNormalDistribution:
        """
        Create a log-normal distribution uncertainty model.

        Args:
            bounds (Tuple[float, float]): The lower and upper bounds for the distribution.

        Returns:
            LogNormalDistribution: The log-normal distribution uncertainty model.
        """
        mu, sigma = self.get_mu_sigma()
        uncertainty_model = LogNormalDistribution(
            self.num_uncertainty_qubits, mu=mu, sigma=sigma**2, bounds=bounds
        )
        return uncertainty_model

    def get_european_call_objective(
        self, bounds: Tuple[float, float], mode: str = "call"
    ) -> LinearAmplitudeFunction:
        """
        Create a linear amplitude function for European call or put options.

        Args:
            bounds (Tuple[float, float]): The lower and upper bounds for the distribution.
            mode (str, optional): The type of option to price, "call" or "put". Default is "call".

        Returns:
            LinearAmplitudeFunction: The linear amplitude function for the option.

        Raises:
            ValueError: If the mode specified is not "call" or "put".
        """
        breakpoints = [bounds[0], self._strike_price]
        if mode == "call":
            slopes = [0, 1]
            offsets = [0, 0]
            f_min = 0
            f_max = bounds[1] - self._strike_price
        elif mode == "put":
            slopes = [-1, 0]
            offsets = [self._strike_price - bounds[0], 0]
            f_min = 0
            f_max = self._strike_price - bounds[0]
        else:
            raise ValueError(f"mode: {mode} must be either 'call' or 'put'.")

        european_call_objective = LinearAmplitudeFunction(
            self.num_uncertainty_qubits,
            slopes,
            offsets,
            domain=bounds,
            image=(f_min, f_max),
            breakpoints=breakpoints,
            rescaling_factor=self.c_approx,
        )
        return european_call_objective

    def to_estimation_problem(
        self, uncertainty_model, european_call_objective
    ) -> EstimationProblem:
        """
        Convert the uncertainty model and objective function into an estimation problem.

        Args:
            uncertainty_model: The log-normal distribution uncertainty model.
            european_call_objective: The linear amplitude function for the option.

        Returns:
            EstimationProblem: The estimation problem for quantum amplitude estimation.
        """
        num_qubits = european_call_objective.num_qubits
        european_call = QuantumCircuit(num_qubits)
        european_call.append(uncertainty_model, range(self.num_uncertainty_qubits))
        european_call.append(european_call_objective, range(num_qubits))
        problem = EstimationProblem(
            state_preparation=european_call,
            objective_qubits=[self.num_uncertainty_qubits],
            post_processing=european_call_objective.post_processing,
        )
        return problem

    def run_amplitude_estimation(self, problem):
        """
        Run the quantum amplitude estimation algorithm.

        Args:
            problem (EstimationProblem): The estimation problem to solve.

        Returns:
            result: The result of the amplitude estimation.
        """
        ae = IterativeAmplitudeEstimation(
            epsilon_target=self.epsilon,
            alpha=self.alpha,
            sampler=Sampler(run_options={"shots": self.shots, "seed": self.ae_seed}),
        )
        result = ae.estimate(problem)
        return result

    def solve(self, mode="call") -> float:
        """
        Solve for the option price using the QMC method based on the specified mode (call or put).

        Args:
            mode (str, optional): The type of option to price, "call" or "put". Default is "call".

        Returns:
            float: The estimated price of the option.
        """
        bounds = self.get_bounds()
        uncertainty_model = self.get_uncertainty_model(bounds)
        european_call_objective = self.get_european_call_objective(bounds, mode)
        problem = self.to_estimation_problem(uncertainty_model, european_call_objective)
        result = self.run_amplitude_estimation(problem)
        conf_int = np.array(result.confidence_interval_processed)
        estimate_result = result.estimation_processed
        real_estimte_result = self.get_real_value(estimate_result)

        # print result
        x = uncertainty_model.values
        if mode == "call":
            y = np.maximum(0, x - self._strike_price)
        elif mode == "put":
            y = np.maximum(0, self._strike_price - x)
        exact_value = np.dot(uncertainty_model.probabilities, y)
        print(
            f"Exact value:\t[{exact_value:.4f}/{self.get_real_value(exact_value):.4f}]"
        )
        print(
            f"Estimated:\t[{estimate_result:.4f}/{self.get_real_value(estimate_result):.4f}]"
        )

        return real_estimte_result

    def get_real_value(self, value: float) -> float:
        """
        Calculate the real value of the estimated option price.

        Args:
            value (float): The estimated value.

        Returns:
            float: The discounted value adjusted for the risk-free rate.
        """
        return value * np.exp(-self._risk_free_rate * self._expiration)


class OptionPricing(object):
    """
    A class to perform European option pricing using different solvers:
    Black-Scholes-Merton (BSM), Classical Monte Carlo (CMC), and Quantum Monte Carlo (QMC).

    Attributes:
        common_params (dict): Parameters of the problem. A dictionary containing the common parameters required by all solvers.
        extra_params (dict): Parameter of algorithm. A dictionary containing additional parameters specific to CMC or QMC solvers.
    """

    def __init__(
        self,
        stock_price: float,
        strike_price: float,
        expiration: float,
        risk_free_rate: float,
        volatility: float,
        **kwargs,
    ) -> None:
        """
        Initialize the OptionPricing class with the given parameters.

        Args:
            stock_price (float): The current price of the stock.
            strike_price (float): The strike price of the option.
            expiration (float): The time to expiration of the option in years.
            risk_free_rate (float): The risk-free interest rate.
            volatility (float): The volatility of the stock.
            kwargs: Additional parameters for CMC or QMC solvers.
        """
        self.common_params = {
            "stock_price": stock_price,
            "strike_price": strike_price,
            "expiration": expiration,
            "risk_free_rate": risk_free_rate,
            "volatility": volatility,
        }
        self.extra_params = kwargs

    def get_ans(self, solver: str = "QMC", mode: str = "call") -> float:
        """
        Calculate the option price using the specified solver.

        Args:
            solver (str): The solver to use for pricing the option. Can be "BSM", "CMC", or "QMC".
            mode (str): The type of option to price. Can be "call" for a call option or "put" for a put option.

        Returns:
            float: The calculated price of the option.

        Raises:
            ValueError: If the solver specified is not "BSM", "CMC", or "QMC".
        """
        if solver == "BSM":
            BSM_solver = BSM(**self.common_params)
            value = BSM_solver.solve(mode)
        elif solver == "CMC":
            CMC_solver = CMC(**self.common_params, **self.extra_params)
            value = CMC_solver.solve(mode)
        elif solver == "QMC":
            QMC_solver = QMC(**self.common_params, **self.extra_params)
            value = QMC_solver.solve(mode)
        else:
            raise ValueError(f"solver: {solver} should be BSM, CMC or QMC.")

        return value


if __name__ == "__main__":
    stock_price = 192.523  # 初始价, starting price of the stock ticker is $100
    strike_price = 198.403  # 行权价：期权合约中约定的可以行使期权的价格
    expiration = 90 / 365  # 到期时间, this means 1 month
    risk_free_rate = 0.044  # 无风险利率, risk free rate is a measure of "opportunity cost." For example, you could choose to invest your money in bonds instead for a safer return of ~4%.
    volatility = 0.199  # 股票的波动性, volatility of the stock. This is a measure of how much the stock price fluctuates. The higher the volatility, the more the stock price fluctuates.

    def test_BSM():
        solver = OptionPricing(
            stock_price, strike_price, expiration, risk_free_rate, volatility
        )
        call_value = solver.get_ans(solver="BSM", mode="call")
        put_value = solver.get_ans(solver="BSM", mode="put")
        print(f"[Call: {call_value}]\n[Put : {put_value}]")

    def test_CMC():
        solver = OptionPricing(
            stock_price, strike_price, expiration, risk_free_rate, volatility
        )
        call_value = solver.get_ans(solver="CMC", mode="call")
        put_value = solver.get_ans(solver="CMC", mode="put")
        print(f"[Call: {call_value}]\n[Put : {put_value}]")

    def test_QMC():
        # Creating an OptionPricing object with additional parameters for QMC
        solver = OptionPricing(
            stock_price,
            strike_price,
            expiration,
            risk_free_rate,
            volatility,
            num_uncertainty_qubits=3,
            bound_param=3.0,
            c_approx=0.03,
            epsilon=0.003,
            alpha=0.05,
            shots=100,
            ae_seed=42,
        )

        # Getting the option price using the QMC solver
        call_value = solver.get_ans(solver="QMC", mode="call")
        put_value = solver.get_ans(solver="QMC", mode="put")
        print(f"[Call: {call_value}]\n[Put : {put_value}]")

    test_BSM()
    test_CMC()
    test_QMC()
