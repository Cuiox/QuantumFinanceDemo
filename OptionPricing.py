import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

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
            num_qubits: 用于离散随机变量的量子比特数.
            mu: 分布的参数 :math:`\mu`.
            sigma: 分布的参数 :math:`\Sigma`.
            bounds: 分布的截断边界.
            upto_diag: 可忽略.
            name: 可忽略.
        """
        inner = QuantumCircuit(num_qubits, name=name)
        x = np.linspace(bounds[0], bounds[1], num=2**num_qubits)
        # 计算规范化的、截断的概率
        probabilities = []
        from scipy.stats import multivariate_normal

        for x_i in x:
            # 从normal到log-normal引用的映射概率
            if np.min(x_i) > 0:
                det = 1 / np.prod(x_i)
                probability = multivariate_normal.pdf(np.log(x_i), mu, sigma) * det
            else:
                probability = 0
            probabilities += [probability]
        normalized_probabilities = probabilities / np.sum(probabilities)

        # 存储为属性
        self._values = x
        self._probabilities = normalized_probabilities
        self._bounds = bounds

        super().__init__(*inner.qregs, name=name)

        # 使用默认等距算法来构建电路
        if upto_diag:
            inner.isometry(np.sqrt(normalized_probabilities), inner.qubits, None)
            self.append(inner.to_instruction(), inner.qubits)
        else:
            from qiskit.extensions import (
                Initialize,
            )

            initialize = Initialize(np.sqrt(normalized_probabilities))
            circuit = initialize.gates_to_uncompute().inverse()
            inner.compose(circuit, inplace=True)
            self.append(inner.to_gate(), inner.qubits)

    @property
    def values(self) -> np.ndarray:
        """返回随机变量的离散点ß."""
        return self._values

    @property
    def probabilities(self) -> np.ndarray:
        """返回值的采样概率."""
        return self._probabilities  # type: ignore

    @property
    def bounds(self) -> Union[Tuple[float, float], List[Tuple[float, float]]]:
        """返回概率分布的边界."""
        return self._bounds


class BSM(object):
    """
    经典方法: 使用BSM模型求解期权定价问题
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
        给定问题的参数并进行初始化

        Args:
            stock_price (float): 股票的初始价格.
            strike_price (float): 期权的行权价.
            expiration (float): 期权的到期时长(以年为单位).
            risk_free_rate (float): 无风险利率.
            volatility (float): 股票的波动性.
        """
        self._stock_price = stock_price
        self._strike_price = strike_price
        self._expiration = expiration
        self._risk_free_rate = risk_free_rate
        self._volatility = volatility

    def d1(self) -> float:
        """
        辅助函数: 计算BSM公式中的d1.

        Returns:
            float: d1的值.
        """
        return (
            log(self._stock_price / self._strike_price)
            + (self._risk_free_rate + self._volatility**2 / 2.0) * self._expiration
        ) / (self._volatility * sqrt(self._expiration))

    def d2(self) -> float:
        """
        辅助函数: 计算BSM公式中的d2.

        Returns:
            float: d2的值.
        """
        return self.d1() - self._volatility * sqrt(self._expiration)

    def bs_call(self) -> float:
        """
        根据BSM公式计算欧式看涨期权的价格.

        Returns:
            float: 看涨期权的价格.
        """
        return self._stock_price * norm.cdf(self.d1()) - self._strike_price * exp(
            -self._risk_free_rate * self._expiration
        ) * norm.cdf(self.d2())

    def bs_put(self) -> float:
        """
        根据BSM公式计算欧式看跌期权的价格.

        Returns:
            float: 看跌期权的价格.
        """
        return (
            norm.cdf(-self.d2())
            * self._strike_price
            * exp(-self._risk_free_rate * self._expiration)
            - norm.cdf(-self.d1()) * self._stock_price
        )

    def solve(self, mode: str = "call") -> float:
        """
        BSM模型求解期权定价问题.

        Args:
            mode (str): 期权的类型, "call" or "put", "看涨"或"看跌".

        Returns:
            float: 期权的价格.

        Raises:
            ValueError: 若 `mode` 并非 "call" or "put".
        """
        # 根据不同的 mode 调用不同的方法.
        if mode == "call":
            return self.bs_call()
        elif mode == "put":
            return self.bs_put()
        else:
            raise ValueError(f"mode: {mode} 必须是 'call' 或 'put'.")


class CMC(object):
    """
    经典方法: 使用蒙特卡罗方法求解期权定价问题
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
        给定问题的参数以及算法的参数, 并进行初始化.

        Args:
            stock_price (float): 股票的初始价格.
            strike_price (float): 期权的行权价.
            expiration (float): 期权的到期时长(以年为单位).
            risk_free_rate (float): 无风险利率.
            volatility (float): 股票的波动性.
            simulation_count (int, optional): 蒙特卡罗模拟的次数.
            trading_days_per_month (int, optional): 每个月的交易天数.
        """
        if simulation_count <= 0:
            raise ValueError("simulation_count 必须是正数.")
        if trading_days_per_month <= 0 or trading_days_per_month > 31:
            raise ValueError("trading_days_per_month 必须在1到31之间.")
        self._stock_price = stock_price
        self._strike_price = strike_price
        self._expiration = expiration
        self._risk_free_rate = risk_free_rate
        self._volatility = volatility
        self.simulation_count = simulation_count
        self.trading_days_per_month = trading_days_per_month
        # trading_days 为总的交易天数
        self.trading_days = int(self._expiration * 12 * self.trading_days_per_month)

    def solve(self, mode: str = "call") -> float:
        """
        使用蒙特卡罗方法求解.

        Args:
            mode (str): 期权的类型, "call" or "put", "看涨"或"看跌".

        Returns:
            float: 期权的价格.

        Raises:
            ValueError: 若 `mode` 并非 "call" or "put".
        """
        # 计算相关系数
        t = self._expiration / self.trading_days
        drift = (self._risk_free_rate - self._volatility**2 / 2) * t
        epsilon = np.random.normal(0, 1, (self.simulation_count, self.trading_days))
        a = self._volatility * np.sqrt(t)

        # 初始化 prices
        prices = np.zeros((self.simulation_count, self.trading_days))
        prices[:, 0] = self._stock_price

        # 进行蒙特卡罗模拟
        for timestep in range(1, self.trading_days):
            prices[:, timestep] = prices[:, timestep - 1] * np.exp(
                drift + a * epsilon[:, timestep]
            )

        # 根据 mode 的不同选择计算看涨期权或看跌期权
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
            raise ValueError(f"mode: {mode} 必须是 'call' 或 'put'.")

        return Value


class QMC(object):
    """
    量子方法: 使用量子蒙特卡罗方法求解期权定价问题
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
        epsilon: float = 0.003,
        alpha: float = 0.05,
        shots: int = 100,
        ae_seed: Optional[int] = 42,
    ) -> None:
        """
        给定问题的参数以及算法的参数, 并进行初始化.

        Args:
            stock_price (float): 股票的初始价格.
            strike_price (float): 期权的行权价.
            expiration (float): 期权的到期时长(以年为单位).
            risk_free_rate (float): 无风险利率.
            volatility (float): 股票的波动性.
            num_uncertainty_qubits (int, optional): 不确定模型的量子比特数.
            bound_param (float, optional): log-normal distribution 的边界参数.
            c_approx (float, optional):支付函数的近似比例因子.
            epsilon (float, optional): 振幅估计的目标精度.
            alpha (float, optional): 振幅估计的信心水平.
            shots (int, optional): 量子算法的模拟次数.
            ae_seed (Optional[int], optional): 振幅估计算法的随机种子.
        """
        if num_uncertainty_qubits > 10 or num_uncertainty_qubits < 1:
            raise ValueError("num_uncertainty_qubits 数值不满足要求, 需在1~10之间.")
        if bound_param <= 0:
            raise ValueError("bound_param 必须是正数.")
        if c_approx <= 0:
            raise ValueError("c_approx 必须是正数.")
        if epsilon <= 0:
            raise ValueError("epsilon 必须是正数.")
        if alpha <= 0:
            raise ValueError("alpha 必须是正数.")
        if shots <= 0:
            raise ValueError("shots 必须是正数.")

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
        计算 log-normal distribution 的参数 mu 和 sigma。

        Returns:
            Tuple[float, float]: 计算得到的 mu 和 sigma 值.
        """
        mu = (
            self._risk_free_rate - 0.5 * self._volatility**2
        ) * self._expiration + np.log(self._stock_price)
        sigma = self._volatility * np.sqrt(self._expiration)
        return mu, sigma

    def get_bounds(self) -> Tuple[float, float]:
        """
        计算 log-normal distribution 的参数 bounds.

        Returns:
            Tuple[float, float]: 分布的上下界.
        """
        # 计算所需的参数
        mu, sigma = self.get_mu_sigma()
        mean = np.exp(mu + sigma**2 / 2)
        variance = (np.exp(sigma**2) - 1) * np.exp(2 * mu + sigma**2)
        stddev = np.sqrt(variance)
        # 计算上下界, 即现货价格所考虑的最低和最高价值
        low = np.maximum(0, mean - self.bound_param * stddev)
        high = mean + self.bound_param * stddev
        bounds = (low, high)
        return bounds

    def get_uncertainty_model(
        self, bounds: Tuple[float, float]
    ) -> LogNormalDistribution:
        """
        创建 log-normal distribution 不确定模型.

        Args:
            bounds (Tuple[float, float]): 分布的上下界.

        Returns:
            LogNormalDistribution: log-normal distribution 不确定模型.
        """
        mu, sigma = self.get_mu_sigma()
        # 调用 LogNormalDistribution 创建不确定模型
        uncertainty_model = LogNormalDistribution(
            self.num_uncertainty_qubits, mu=mu, sigma=sigma**2, bounds=bounds
        )
        return uncertainty_model

    def get_european_call_objective(
        self, bounds: Tuple[float, float], mode: str = "call"
    ) -> LinearAmplitudeFunction:
        """
        创建期权的线性振幅函数.

        Args:
            bounds (Tuple[float, float]): 分布的上下界.
            mode (str, optional): 期权的类型, "call" or "put".

        Returns:
            LinearAmplitudeFunction: 期权的线性振幅函数.

        Raises:
            ValueError: 若 `mode` 并非 "call" or "put".
        """
        # 根据期权的类型, 分别设置参数
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

        # 调用 LinearAmplitudeFunction 以构建
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
        将不确定性模型和目标函数转换为估计问题.

        Args:
            uncertainty_model: log-normal distribution 不确定模型.
            european_call_objective: 期权的线性振幅函数.

        Returns:
            EstimationProblem: 量子振幅估计的估计问题.
        """
        # 获取量子比特数
        num_qubits = european_call_objective.num_qubits
        # 初始化
        european_call = QuantumCircuit(num_qubits)
        # 添加不确定模型
        european_call.append(uncertainty_model, range(self.num_uncertainty_qubits))
        # 添加目标函数
        european_call.append(european_call_objective, range(num_qubits))
        # 转化为估计问题
        problem = EstimationProblem(
            state_preparation=european_call,
            objective_qubits=[self.num_uncertainty_qubits],
            post_processing=european_call_objective.post_processing,
        )
        return problem

    def run_amplitude_estimation(self, problem):
        """
        运行量子振幅估计算法.

        Args:
            problem (EstimationProblem): 待解决的估计问题.

        Returns:
            result: 振幅估计的结果.
        """
        # 采用 Iterative Amplitude Estimation 算法
        ae = IterativeAmplitudeEstimation(
            epsilon_target=self.epsilon,
            alpha=self.alpha,
            sampler=Sampler(run_options={"shots": self.shots, "seed": self.ae_seed}),
        )
        # 运行该算法
        result = ae.estimate(problem)
        return result

    def solve(self, mode="call") -> float:
        """
        根据指定的模式(看涨或看跌)使用QMC方法求解期权价格.

        Args:
            mode (str, optional): 期权的类型, "call" or "put".

        Returns:
            float: 期权的估计价格.
        """
        # 获取不确定模型的边界
        bounds = self.get_bounds()
        # 构建不确定模型
        uncertainty_model = self.get_uncertainty_model(bounds)
        # 构建目标函数
        european_call_objective = self.get_european_call_objective(bounds, mode)
        # 转化为估计问题
        problem = self.to_estimation_problem(uncertainty_model, european_call_objective)
        # 运行振幅估计算法
        result = self.run_amplitude_estimation(problem)
        # 获取结果
        estimate_result = result.estimation_processed
        # 将结果进行贴现获取真正的期权价格
        real_estimte_result = self.get_real_value(estimate_result)

        return real_estimte_result

    def get_real_value(self, value: float) -> float:
        """
        计算期权价格的真实值.

        Args:
            value (float): 估计值.

        Returns:
            float: 根据无风险利率调整后的贴现价值.
        """
        return value * np.exp(-self._risk_free_rate * self._expiration)


class OptionPricing(object):
    """
    使用不同的模型解决欧式期权定价问题:
    Black-Scholes-Merton (BSM), Classical Monte Carlo (CMC), and Quantum Monte Carlo (QMC).

    Attributes:
        common_params (dict): 期权定价问题的参数.
        extra_params (dict): 额外的与算法相关的参数.
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
        对参数进行检查并初始化.

        Args:
            stock_price (float): 股票的初始价格.
            strike_price (float): 期权的行权价.
            expiration (float): 期权的到期时长(以年为单位).
            risk_free_rate (float): 无风险利率.
            volatility (float): 股票的波动性.
            kwargs: 额外的参数.
        """
        # 检查 stock_price 和 strike_price
        if stock_price <= 0:
            raise ValueError("Stock price 必须是正数.")
        if strike_price <= 0:
            raise ValueError("Strike price 必须是正数.")
        # 检查 expiration
        if expiration <= 0:
            raise ValueError("Expiration 必须是正数.")
        # 检查 risk_free_rate
        if risk_free_rate < 0:
            raise ValueError("Risk-free rate 必须是非负数.")
        # 检查 volatility
        if volatility <= 0:
            raise ValueError("Volatility 必须是正数.")

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
        利用指定的模型解决期权定价问题.

        Args:
            solver (str): 用于求解的模型, 可以是 "BSM", "CMC", 或 "QMC".
            mode (str): 期权的类型, "call" or "put", "看涨"或"看跌".

        Returns:
            float: 期权的价格.

        Raises:
            ValueError: 若 solver 并非 "BSM", "CMC", 或 "QMC".
        """
        # 根据不同的 solver 调用不同的模型.
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
    stock_price = 192.523  # 初始价
    strike_price = 198.403  # 行权价
    expiration = 90 / 365  # 到期时间
    risk_free_rate = 0.044  # 无风险利率
    volatility = 0.199  # 股票的波动性

    def test_BSM():
        print(f"BSM")
        solver = OptionPricing(
            stock_price, strike_price, expiration, risk_free_rate, volatility
        )
        call_value = solver.get_ans(solver="BSM", mode="call")
        put_value = solver.get_ans(solver="BSM", mode="put")
        print(f"[Call: {call_value}]\n[Put : {put_value}]")

    def test_CMC():
        print(f"CMC")
        solver = OptionPricing(
            stock_price, strike_price, expiration, risk_free_rate, volatility
        )
        call_value = solver.get_ans(solver="CMC", mode="call")
        put_value = solver.get_ans(solver="CMC", mode="put")
        print(f"[Call: {call_value}]\n[Put : {put_value}]")

    def test_QMC():
        print(f"QMC")
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

        call_value = solver.get_ans(solver="QMC", mode="call")
        put_value = solver.get_ans(solver="QMC", mode="put")
        print(f"[Call: {call_value}]\n[Put : {put_value}]")

    try:
        test_BSM()
        test_CMC()
        test_QMC()
    except ValueError as e:
        print(e)
