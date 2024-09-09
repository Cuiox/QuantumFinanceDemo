# Improving Scalability and Accuracy for Portfolio Optimization
from __future__ import annotations

import pickle
from typing import List, Optional

import numpy as np
import scipy.optimize as opt
from sklearn import svm

import json
import math
import random

import pandas as pd

from typing import Callable

import pandas as pd

from abc import ABCMeta


class LocalFileLoader(metaclass=ABCMeta):
    def __init__(
        self,
        src: str = None,
        file_type: str = "excel",
        read_func: Callable | None = None,
    ) -> None:
        """Load data from local file.

        Args:
            src: Local file's path.
            file_type: Currently support csv and excel for automatic loading.
            read_func: If is not `None`, use this function to customize data loading.
        """
        # super().__init__(src)
        self._data_source = src if src is not None else "resources/Nasdaq-100-2022.xlsx"
        self.__read = read_func
        if self.__read is None:
            if file_type == "csv":
                self.__read = pd.read_csv
            elif file_type == "excel":
                self.__read = pd.read_excel
            else:
                raise RuntimeError("Unsupported file type.")

    def load(self, post_process: Callable | None = None):
        assert self.__read is not None
        data = self.__read(self._data_source)
        if isinstance(post_process, Callable):
            data = post_process(data)
        return data


class StockDataLoader(object):
    def __init__(
        self,
        data_src: LocalFileLoader | None = None,
        valid_tickers_path: str | None = None,
        tickers_pairs: list[list[str]] | list[str] | None = None,
        num_pairs: int | None = None,
        num_assets: int | None = None,
    ) -> None:
        """
        Args:
            valid_tickers_path: File path to valid tickers.
            num_pairs: The number of stock pairs.
            num_assets: The number of assets in the stock portfolio.
        """
        super().__init__()
        self.data_src = data_src if data_src is not None else LocalFileLoader()
        self.valid_tickers_path = valid_tickers_path
        self.stock_data = self.load_stock_data()
        if isinstance(tickers_pairs, list):
            if all(isinstance(ticker, str) for ticker in tickers_pairs):
                # Convert list[str] to list[list[str]]
                self.tickers_pairs = [tickers_pairs]
                if num_pairs is not None:
                    if num_pairs == 1:
                        self.num_pairs = num_pairs
                    else:
                        raise ValueError(
                            "The given parameter num_pairs is inconsistent with tickers_pairs"
                        )
                else:
                    self.num_pairs = 1

                if num_assets is not None:
                    if num_assets == len(tickers_pairs):
                        self.num_assets = num_assets
                    else:
                        raise ValueError(
                            "The given parameter num_assets is inconsistent with tickers_pairs"
                        )
                else:
                    self.num_assets = len(tickers_pairs)

            elif all(isinstance(tickers_pair, list) for tickers_pair in tickers_pairs):
                # Ensure all lists in tickers_pairs have the same length
                pair_lengths = {len(tickers_pair) for tickers_pair in tickers_pairs}
                if len(pair_lengths) > 1:
                    raise ValueError("All ticker pairs must have the same length.")

                self.tickers_pairs = tickers_pairs
                if num_pairs is not None:
                    if num_pairs == len(tickers_pairs):
                        self.num_pairs = num_pairs
                    else:
                        raise ValueError(
                            "The given parameter num_pairs is inconsistent with tickers_pairs"
                        )
                else:
                    self.num_pairs = len(tickers_pairs)

                if num_assets is not None:
                    if num_assets == len(tickers_pairs[0]):
                        self.num_assets = num_assets
                    else:
                        raise ValueError(
                            "The given parameter num_assets is inconsistent with tickers_pairs"
                        )
                else:
                    self.num_assets = len(tickers_pairs[0])
            else:
                raise ValueError(
                    "tickers_pairs must be a list of strings or a list of list of strings."
                )
        elif tickers_pairs is None:
            self.num_pairs = num_pairs if num_pairs is not None else 1
            self.num_assets = num_assets if num_assets is not None else 3
            self.tickers_pairs = self.gen_tickers_pairs()
        else:
            raise TypeError(
                "tickers_pairs must be a list of strings, a list of list of strings, or None."
            )

    def get_stock_data(self):
        return self.stock_data

    def get_tickers_pairs(self) -> list[list[str]]:
        return self.tickers_pairs

    def load_stock_data(self):
        return self.data_src.load()

    def gen_tickers_pairs(self) -> list[list[str]]:
        """Get num_pairs num_assets tickers pairs.
        Return:
            list: A list containing 'num_pairs' of unique stock ticker pairs
        """

        def get_valid_US_tickers(
            valid_tickers_path: str = "resources/valid_amex_tickers_v2.json",
        ):
            tickers_path = valid_tickers_path
            with open(tickers_path) as f:
                ticker_symbols = json.load(f)
            return ticker_symbols

        # TODO: 传入数组过滤，若valid为空则默认所有都可以。
        valid_tickers = get_valid_US_tickers(self.valid_tickers_path)
        # NOTE: 如果要求的组合数过多，则取小的。
        combinations = math.comb(len(valid_tickers), self.num_assets)
        if combinations < self.num_pairs:
            self.num_pairs = combinations
            print(
                f"num_pairs [{self.num_pairs}] exceeds combinations [{combinations}]. Adjusting to {combinations}."
            )  # or Raise ValueError?

        data = []
        while len(data) < self.num_pairs:
            pair = random.sample(valid_tickers, self.num_assets)
            pair.sort()
            if pair not in data:
                data.append(pair)

        return data

    def get_data_et_pairs(self):
        pass


from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np
import scipy as sp

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import PhaseEstimation
from qiskit.quantum_info import Statevector
from qiskit.opflow import I, StateFn, TensoredOp, Z


class AbstractSolver(ABC):
    def __init__(self, matrix: List | np.ndarray, vector: List | np.ndarray) -> None:
        self.matrix = matrix
        self.vector = vector

    def _print_complexity(self) -> None:
        print(f"matrix size: {np.shape(self.matrix)}.")
        print(f"kappa: {np.linalg.cond(self.matrix)}.")
        print(f"lambda: {np.linalg.eigvals(self.matrix)}")

    @abstractmethod
    def solve(self) -> np.ndarray:
        self._print_complexity()


class NumpySolver(AbstractSolver):
    def __init__(self, matrix, vector) -> None:
        super().__init__(matrix, vector)

    def solve(self):
        return np.linalg.solve(self.matrix, self.vector)


class HHLSolver(AbstractSolver):
    def __init__(
        self,
        matrix,
        vector,
        lambda_min,
        lambda_max,
        *,
        phase_qubit_num: int = None,
        epsilon: float = 1 / 32,
        disturbance: Tuple[float, float] = (0, 0),
    ) -> None:
        self._phase_qubit_num = None
        self.matrix = np.array(matrix)
        self.norm_const = 1
        self.scaling = np.linalg.norm(vector)
        self.vector = vector / np.linalg.norm(vector)
        self.vector_qubit_num = int(np.log2(np.shape(self.vector)))
        self._phase_qubit_num = phase_qubit_num
        self.epsilon = epsilon
        self.disturbance = disturbance
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

    @property
    def phase_qubit_num(self) -> int:
        if self._phase_qubit_num is None:
            abs_lambda_min = self.lambda_min * (1 + self.disturbance[0])
            abs_lambda_max = self.lambda_max * (1 + self.disturbance[1])
            condition_number = abs_lambda_max / abs_lambda_min
            return int(2 + np.ceil(np.log2(condition_number / self.epsilon)))

        return self._phase_qubit_num

    def _get_parameter_values(self) -> None:
        abs_lambda_min = self.lambda_min * (1 + self.disturbance[0])
        abs_lambda_max = self.lambda_max * (1 + self.disturbance[1])
        lambda_scaling = 0.5 / abs_lambda_max
        self.scaling *= lambda_scaling
        self.matrix *= lambda_scaling
        abs_lambda_min *= lambda_scaling
        abs_lambda_max *= lambda_scaling

        if self.phase_qubit_num is None:
            print("It is impossible.")

        self.norm_const = abs_lambda_min

    def _construct_rotation(self, n_state_qubits: int):
        reg_state = QuantumRegister(n_state_qubits, "state")
        reg_flag = QuantumRegister(1, "flag")
        circuit = QuantumCircuit(reg_state, reg_flag, name="UCRY")
        angles = [0]
        tot = 2**n_state_qubits
        for i in range(1, tot):
            phi = i / tot
            rotation_value = (
                self.norm_const * 0.5 / (phi - (i >= 2 ** (n_state_qubits - 1)))
            )
            if np.isclose(rotation_value, 1, 1e-5, 1e-5):
                angles.append(np.pi)
            elif np.isclose(rotation_value, -1):
                angles.append(-np.pi)
            elif -1 < rotation_value < 1:
                angles.append(2 * np.arcsin(rotation_value))
            else:
                angles.append(0)
        circuit.ucry(angles, reg_state[:], reg_flag[:])
        return circuit

    def construct_circuit(self, need_measurement=False):
        self._get_parameter_values()
        reg_s = QuantumRegister(int(np.log2(np.shape(self.vector))), name="vector")
        reg_r = QuantumRegister(self.phase_qubit_num, "phases")
        reg_a = QuantumRegister(1, name="flag")

        vector_circuit = QuantumCircuit(reg_s.size, name="isometry")
        vector_circuit.iso(self.vector, vector_circuit.qubits, None)

        matrix_circuit = QuantumCircuit(reg_s.size, name="U")
        matrix_circuit.unitary(
            sp.linalg.expm(1j * self.matrix * np.pi),
            matrix_circuit.qubits,
        )
        phase_estimation = PhaseEstimation(reg_r.size, matrix_circuit)
        reciprocal_circuit = self._construct_rotation(reg_r.size)
        circuit = QuantumCircuit(reg_s, reg_r, reg_a)
        circuit.append(vector_circuit, reg_s[:])
        circuit.append(phase_estimation, reg_r[:] + reg_s[:])
        circuit.append(reciprocal_circuit, reg_r[::-1] + reg_a[:])
        circuit.append(phase_estimation.inverse(), reg_r[:] + reg_s[:])

        if need_measurement is True:
            reg_measurement = ClassicalRegister(1, "measure")
            circuit.add_register(reg_measurement)
            circuit.measure(reg_a[:], reg_measurement[:])

        return circuit

    def solve(self, circuit=None):
        if circuit is None:
            circuit = self.construct_circuit()
        ss = Statevector(circuit)
        statevector = ss.data
        statevector_real = np.real(statevector)
        probability = ss.probabilities_dict()
        success_probability = 0
        for key, value in probability.items():
            if key[0] == "1":
                success_probability += value
        norm = np.real(np.sqrt(success_probability) / self.norm_const)

        state = statevector_real[
            [
                int(
                    "1"
                    + circuit.qregs[1].size * "0"
                    + np.binary_repr(i, width=circuit.qregs[0].size),
                    2,
                )
                for i in range(2 ** circuit.qregs[0].size)
            ]
        ]

        return self.scaling * state * norm / np.linalg.norm(state)

    def _calculate_norm(self, qc: QuantumCircuit) -> float:
        nb = qc.qregs[0].size
        nl = qc.qregs[1].size
        zero_op = (I + Z) / 2
        one_op = (I - Z) / 2
        observable = one_op ^ TensoredOp(nl * [zero_op]) ^ (I ^ nb)
        success_probability = (~StateFn(observable) @ StateFn(qc)).eval()
        return np.real(np.sqrt(success_probability) / self.norm_const)


def get_RSP(stock_data, tickers_pair) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the expected returns (R), covariance matrix (S), and a unit vector (P) for a pair of tickers.

    Args:
        stock_data: A dictionary containing stock data.
        tickers_pair: A list of ticker symbols.

    Returns:
        tuple: A tuple containing the expected returns vector (R), covariance matrix (S), and unit vector (P).
    """
    value_matrix = np.array(
        [stock_data[ticker_code]["Adj Close"] for ticker_code in tickers_pair]
    )
    period_return = value_matrix[:, 1:] / value_matrix[:, :-1] - 1
    num_assets = len(tickers_pair)
    return (
        np.mean(period_return, axis=1),  # R
        np.cov(period_return, ddof=1),  # \Sigma
        np.ones(num_assets),  # \Pi
    )


def set_random(
    seed_value: int = 42,
) -> None:
    """Set the random seed for reproducibility."""
    np.random.seed(seed_value)
    random.seed(seed_value)


class PortfolioConstructor:
    def __init__(
        self,
        income_mean: np.ndarray | List,
        income_cov: np.ndarray | List,
        price: Optional[np.array | List],
        s1: float = 1,
        s2: float = 1,
        s3: float = 1,
        budget: float = 1,
        expected_income: float = 1,
        solver: Optional[str] = "hhl",
    ) -> None:
        """Initialize the PortfolioConstructor.

        Args:
            income_mean: The expected returns vector (R).
            income_cov: The covariance matrix of returns (S).
            price: The price vector (P) for the assets.
            s1: Scaling factor for the expected returns.
            s2: Scaling factor for the prices.
            s3: Scaling factor for the covariance matrix.
            budget: The total budget for the portfolio.
            expected_income: The target expected income from the portfolio.
            solver: The method used for solving the linear equations, either "hhl" or "numpy".

        Raises:
            ValueError: If the input `income_mean` is not a 1D vector or `income_cov` is not a 2D matrix.

        TODO:
            - 直接给 stock_data et tickers_pair, 而不是 R, S, P
            - 用 s 而不是 s1, s2, s3
        """
        if np.ndim(income_mean) != 1:
            raise ValueError("Input `income_mean` must be a one-dimensional vector!")
        if np.ndim(income_cov) != 2:
            raise ValueError("Input `income_cov` must be a two-dimensional matrix!")

        self.num_assets = np.shape(income_mean)[0]
        if isinstance(income_mean, list):
            income_mean = np.array(income_mean)
        if isinstance(income_cov, list):
            income_cov = np.array(income_cov)

        if price is None:
            price = np.ones(self.num_assets)

        self.income_mean = income_mean
        self.income_cov = income_cov
        self.price = price
        self.s1 = s1
        self.s2 = s2
        self.s3 = s3
        self.budget = budget
        self.expected_income = expected_income

        self._matrix = None
        self._vector = None
        self._solver = solver

    def _construct_linear_equation(self) -> Tuple[np.ndarray, np.ndarray]:
        """Construct the matrix and vector for the linear equation.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The constructed matrix and vector.
        """
        return self._construct_matrix(), self._construct_vector()

    def _construct_matrix(self) -> np.ndarray:
        """Construct the coefficient matrix (A) for the linear equation.

        Returns:
            np.ndarray: The constructed matrix A.

        Raises:
            ValueError: If the constructed matrix is not symmetric in the HHL solver case.
        """
        line1 = np.concatenate(([0] * 2, self.income_mean * self.s1), axis=0)
        line2 = np.concatenate(([0] * 2, self.price * self.s2), axis=0)
        line3 = np.concatenate(
            (
                np.reshape(self.income_mean * self.s1, (-1, 1)),
                np.reshape(self.price * self.s2, (-1, 1)),
                self.income_cov,
            ),
            axis=1,
        )

        A = np.concatenate(
            (np.reshape(line1, (1, -1)), np.reshape(line2, (1, -1)), line3),
            axis=0,
        )

        if self._solver == "numpy":
            return A
        elif self._solver == "hhl":
            initial_size = np.shape(A)[0]
            modified_size = int(2 ** np.ceil(np.log2(initial_size)))
            Ca = np.r_[
                np.c_[A, np.zeros((initial_size, modified_size - initial_size))],
                np.c_[
                    np.zeros((modified_size - initial_size, initial_size)),
                    np.identity(modified_size - initial_size) * self.s3,
                ],
            ]

            if not np.all(Ca == Ca.T):
                raise ValueError("Crashed!")
            else:
                return Ca
        else:
            raise ValueError("No such method!")

    def _construct_vector(self) -> np.ndarray:
        """Construct the vector (b) for the linear equation.

        Returns:
            np.ndarray: The constructed vector b.
        """
        b = np.concatenate(
            (
                [self.expected_income * self.s1, self.budget * self.s2],
                np.zeros(self.num_assets),
            ),
            axis=0,
        )
        if self._solver == "numpy":
            return b
        elif self._solver == "hhl":
            initial_size = np.shape(b)[0]
            modified_size = int(2 ** np.ceil(np.log2(initial_size)))
            Cb = np.r_[b, np.zeros(modified_size - initial_size)]
            return Cb

    def _print_info(self, matrix: np.ndarray):
        print(f"asset num: {self.num_assets}.")
        print(f"matrix size: {np.shape(matrix)}.")
        print(f"kappa: {np.linalg.cond(matrix)}.")
        print(f"lambda: {np.linalg.eigvals(matrix)}.")

    @property
    def matrix(self) -> np.ndarray:
        """Property to lazily construct and return the matrix.

        Returns:
            np.ndarray: The matrix for the portfolio optimization problem.
        """
        if self._matrix is None:
            self._matrix = self._construct_matrix()
        return self._matrix

    @property
    def vector(self) -> np.ndarray:
        """Property to lazily construct and return the vector.

        Returns:
            np.ndarray: The vector for the portfolio optimization problem.
        """
        if self._vector is None:
            self._vector = self._construct_vector()
        return self._vector

    @property
    def equation(self) -> Tuple[np.ndarray, np.ndarray]:
        """Property to return the full equation (matrix and vector).

        Returns:
            Tuple[np.ndarray, np.ndarray]: The matrix and vector of the linear equation.
        """
        return self.matrix, self.vector


class SAPO(object):
    def __init__(
        self,
        data_loader: StockDataLoader | None = None,
        scale_model: ScaleModel | None = None,
        lambda_model_min: SVRModel | None = None,
        lambda_model_max: SVRModel | None = None,
    ) -> None:
        """
        Initialize the SAPO class with optional models and data loader.

        Args:
            data_loader (StockDataLoader, optional): An instance of StockDataLoader to provide stock data and tickers pairs.
            scale_model (ScaleModel, optional): An instance of ScaleModel to provide scaling factors.
            lambda_model_min (SVRModel, optional): An SVR model to predict the minimum lambda value.
            lambda_model_max (SVRModel, optional): An SVR model to predict the maximum lambda value.
        """
        self.__data_loader = (
            data_loader if data_loader is not None else StockDataLoader()
        )
        self.__scale_model = scale_model if scale_model is not None else ScaleModel()
        self.__lambda_model_min = (
            lambda_model_min
            if lambda_model_min is not None
            else SVRModel(lambda_type="min")
        )
        self.__lambda_model_max = (
            lambda_model_max
            if lambda_model_max is not None
            else SVRModel(lambda_type="max")
        )

    def get_ans(
        self, phase_qubit_num: int | None = None, epsilon: float = 1 / 16
    ) -> List[np.ndarray]:
        """
        Solve a set of data using the configured models and return the results.

        Args:
            phase_qubit_num (int, optional): The number of phase qubits to use in the solver. If None, it will be computed.
            epsilon (float, optional): The precision parameter for the solver. Defaults to 1/16.

        Returns:
            List[np.ndarray]: A list of results for each ticker pair, with each result being an array of computed values.
        """
        sapo_results = []
        stock_data = self.__data_loader.get_stock_data()
        tickers_pairs = self.__data_loader.get_tickers_pairs()
        the_s = self.__scale_model.get_scale()
        num_assets = self.__data_loader.num_assets
        for tickers_pair in tickers_pairs:
            R, S, P = get_RSP(stock_data, tickers_pair)
            A, b = PortfolioConstructor(R, S, P, *the_s).equation
            lambda_min = self.__lambda_model_min.predict(stock_data, tickers_pair)
            lambda_max = self.__lambda_model_max.predict(stock_data, tickers_pair)
            res = self.solve(
                A, b, lambda_min, lambda_max, num_assets, phase_qubit_num, epsilon
            )
            sapo_results.append(res)

        return sapo_results

    def solve(
        self,
        A: np.ndarray,
        b: np.ndarray,
        lambda_min: float,
        lambda_max: float,
        num_assets: int,
        phase_qubit_num: Optional[int] = None,
        epsilon: float = 1 / 16,
    ) -> np.ndarray:
        """
        Solve the optimization problem for a single set of data.

        Args:
            A (np.ndarray): The matrix A for the portfolio optimization problem.
            b (np.ndarray): The vector b for the portfolio optimization problem.
            lambda_min (float): The minimum lambda value for the problem.
            lambda_max (float): The maximum lambda value for the problem.
            num_assets (int): The number of assets in the portfolio.
            phase_qubit_num (int, optional): The number of phase qubits to use. If None, it will be computed.
            epsilon (float, optional): The precision parameter for the solver. Defaults to 1/16.

        Returns:
            np.ndarray: The result of the optimization for the given data.
        """
        if phase_qubit_num is None:
            cond = lambda_max / lambda_min
            phase_qubit_num = int(2 + np.ceil(np.log2(cond / epsilon)))
        else:
            # Use fixed phase qubit number
            pass
        sapo_res = HHLSolver(
            A,
            b,
            lambda_min,
            lambda_max,
            phase_qubit_num=phase_qubit_num,
            epsilon=epsilon,
        ).solve()[2 : 2 + num_assets]
        return sapo_res


class ScaleModel(object):
    def __init__(
        self,
        mode: str = "fixe",
        stock: str = "amex",
        data_loader: StockDataLoader | None = None,
        s: tuple | None = None,
    ) -> None:
        """Get scale s.
        Case 1: choose from __fixe_scales
        Case 2: calculate the best scale
        Case 3: use the given scale

        Args:
            mode: "fixe" or "best".
            stock: stock name.
            data_loader: .
            s: the given s.
        """
        """
        Initialize the ScaleModel to either use a fixed scale, compute the best scale, or use a provided scale.

        Args:
            mode (str): The mode for scale determination. Can be "fixe" for using predefined scales or "best" for computing the optimal scale. Defaults to "fixe".
            stock (str): The stock name. Valid options are "amex", "nyse", and "nasdaq". Defaults to "amex".
            data_loader (Optional[StockDataLoader], optional): An instance of StockDataLoader to provide stock data and tickers pairs. Used only if mode is "best". Defaults to None.
            s (Optional[tuple], optional): A tuple representing a given scale. If provided, this scale is used directly. Defaults to None.
        """
        self.__mode = mode
        self.__stock = stock
        self.__fixe_scales = {
            "amex": (0.28137361, -0.00066401, -0.0013469),
            "nyse": (0.24827821, 0.00032029, -0.00083963),
            "nasdaq": (0.55690167, 0.00192067, -0.00137207),
        }
        if s is not None:
            self._s = s
        else:
            if self.__mode == "fixe":
                self._s = self.__fixe_scales[self.__stock]
            elif self.__mode == "best":
                self.__data_loader = (
                    data_loader if data_loader is not None else StockDataLoader()
                )
                self._s = self._get_best_s()
            else:
                self._s = (0, 0, 0)

    def get_scale(self) -> tuple:
        """
        Retrieve the scale.

        Returns:
            tuple: The scale values as a tuple (s1, s2, s3).
        """
        return self._s

    def _get_best_s(self) -> tuple:
        """
        Compute the optimal scale values that minimize the condition number of the matrix for given stock data.

        Returns:
            tuple: The computed optimal scale values (s1, s2, s3).
        """

        def get_condition_number(s):
            return np.mean(
                [np.linalg.cond(PortfolioConstructor(*e, *s).matrix) for e in data]
            )

        data = []
        stock_data = self.__data_loader.get_stock_data()
        tickers_pairs = self.__data_loader.get_tickers_pairs()
        for tickers_pair in tickers_pairs:
            R, S, P = get_RSP(stock_data, tickers_pair)
            data.append((R, S, P))

        init_s = (1, 1, -0.001)
        res = opt.minimize(
            fun=get_condition_number,
            x0=init_s,
            method="Nelder-Mead",
        )
        return res.x


class SVRModel(object):
    def __init__(
        self,
        mode: str = "load",
        data_loader: StockDataLoader | None = None,
        scale_model: ScaleModel | None = None,
        model_filename: str | None = None,
        lambda_type: str = "min",
        matrix_scale: int = 1,
    ) -> None:
        """
        Initialize the SVRModel for loading, training, or predicting lambda values.

        Args:
            mode (str): Operation mode of the model. Can be "load" to load an existing model or "train" to train a new model. Defaults to "load".
            data_loader (Optional[StockDataLoader]): An instance of StockDataLoader to provide stock data and tickers pairs. Defaults to None.
            scale_model (Optional[ScaleModel]): An instance of ScaleModel to obtain scaling parameters. Defaults to None.
            model_filename (Optional[str]): The filename(full path) for loading or saving the SVR model. Defaults to None.
            lambda_type (str): Specifies the type of lambda value to predict. Can be "min" for minimum eigenvalue or "max" for maximum eigenvalue. Defaults to "min".
            matrix_scale (Optional[int]): Scale parameter for the matrix used in SVR. Defaults to None.
        """
        self.__mode = mode
        self.__data_loader = (
            data_loader if data_loader is not None else StockDataLoader()
        )
        self.__scale_model = scale_model if scale_model is not None else ScaleModel()
        self.__model_filename = model_filename
        self.__svr_matrix_scale = {
            "min": {"amex": 4000, "nyse": 5000, "nasdaq": 3000},
            "max": {"amex": 400, "nyse": 800, "nasdaq": 100},
        }
        self._lambda_type = lambda_type
        self._matrix_scale = matrix_scale

        if self.__mode == "load":
            self.__svr_model = self.load_svr_model(self.__model_filename)
        elif self.__mode == "train":
            self.__svr_model = self.train_and_load()
        else:
            pass

    def get_svr_model(self):
        """
        Retrieve the SVR model instance.

        Returns:
            The SVR model instance.
        """
        return self.__svr_model

    def predict(self, stock_data, tickers_pair) -> float:
        """
        Predict the lambda value for a given pair of tickers.

        Args:
            tickers_pair (list): List of ticker symbols for which to predict the lambda value.

        Returns:
            float: The predicted lambda value.
        """

        def construct_svr_data(num_assets, R, S, matrix_scale, s):
            """
            Construct the feature matrix for the SVR model.

            Args:
                num_assets (int): The number of assets.
                R (np.ndarray): Vector of expected returns.
                S (np.ndarray): Covariance matrix of returns.
                matrix_scale (int): Scale for the matrix.
                s (tuple): Scale parameters.

            Returns:
                np.ndarray: The constructed feature matrix.
            """
            length = int(num_assets + (num_assets**2 + num_assets) // 2)
            X = np.empty((0, length))
            R_array = R * matrix_scale * s[0]
            upper_S = S[np.triu_indices(S.shape[0])]
            sample = np.concatenate((R_array, upper_S))
            X = np.vstack((X, sample))
            return X

        # stock_data = self.__data_loader.get_stock_data()
        the_s = self.__scale_model.get_scale()
        R, S, P = get_RSP(stock_data, tickers_pair)
        svr_model = self.get_svr_model()
        X = construct_svr_data(len(tickers_pair), R, S, self._matrix_scale, the_s)
        lambda_m = svr_model.predict(X)
        return np.abs(lambda_m[0] / self._matrix_scale)

    def load_svr_model(self, model_filename: str | None = None):
        """
        Load the SVR model from a file.

        Args:
            model_filename (Optional[str]): The filename of the model to load. If not provided, uses the instance's filename.

        Returns:
            The loaded SVR model.
        """
        filename = (
            model_filename if model_filename is not None else self.__model_filename
        )
        assert filename is not None
        with open(filename, "rb") as file:
            svr_model = pickle.load(file)
        return svr_model

    def train_and_load(self):
        """
        Train a SVR model and load it.

        Returns:
            The loaded SVR model.
        """
        model_filename = self.train_svr_model()
        return self.load_svr_model(model_filename)

    def train_svr_model(self) -> str:
        """
        Train a SVR model using the available stock data and tickers pairs.

        Returns:
            str: The filename where the trained SVR model is saved.
        """
        the_s = self.__scale_model.get_scale()
        stock_data = self.__data_loader.get_stock_data()
        tickers_pairs = self.__data_loader.get_tickers_pairs()
        labels = []
        r_list = []
        s_list = []
        for tickers_pair in tickers_pairs:
            R, S, P = get_RSP(stock_data, tickers_pair)
            s_matrix = PortfolioConstructor(R, S, P, *the_s).matrix * self._matrix_scale
            abs_eigenvalues = np.abs(np.linalg.eigvals(s_matrix))
            if self._lambda_type == "min":
                m_abs = np.min(abs_eigenvalues)
            else:
                m_abs = np.max(abs_eigenvalues)
            labels.append(m_abs)
            r_list.append(R * the_s[0] * self._matrix_scale)
            s_list.append(S * self._matrix_scale)

        Y = np.array(labels)
        r_array = np.array(r_list)
        s_array = np.array(s_list)
        num_assets = len(tickers_pairs[0])
        length = int(num_assets + (num_assets**2 + num_assets) // 2)
        X = np.empty((0, length))
        for i in range(len(r_array)):
            S = s_array[i]
            upper_S = S[np.triu_indices(S.shape[0])]
            sample = np.concatenate((r_array[i], upper_S))
            X = np.vstack((X, sample))
        # Train data
        model = svm.SVR(C=10, kernel="rbf", gamma="auto", epsilon=0.1, tol=0.001)
        model.fit(X, Y)
        # file_name = f"resources/model/svr/{stock}_{num_assets}_{typ}.pkl"
        file_name = self.__model_filename  # f"resources/model/svr/{}"
        with open(file_name, "wb") as file:
            pickle.dump(model, file)
        print(f"save lambda model in {file_name}")
        return file_name


class PortfolioOptimization(object):
    """
    A class for optimizing a stock portfolio using the SAPO solver.

    Attributes:
        __data_loader (StockDataLoader): The data loader instance responsible for loading stock data.
        __scale_model (ScaleModel): The model responsible for providing scale.
        __lambda_model_min (SVRModel): The SVR model for the minimum lambda.
        __lambda_model_max (SVRModel): The SVR model for the maximum lambda.
    """

    # TODO: add mode
    def __init__(
        self,
        data_loader: StockDataLoader | None = None,
        scale_model: ScaleModel | None = None,
        lambda_model_min: SVRModel | None = None,
        lambda_model_max: SVRModel | None = None,
    ) -> None:
        """
        Initializes the PortfolioOptimization class with the given parameters.

        Args:
            data_loader (StockDataLoader, optional): A custom data loader instance.
            scale_model (ScaleModel, optional): A custom scale model instance.
            lambda_model_min (SVRModel, optional): A custom SVR model for minimum lambda.
            lambda_model_max (SVRModel, optional): A custom SVR model for maximum lambda.
        """
        self.__data_loader = (
            data_loader if data_loader is not None else StockDataLoader()
        )
        self.__scale_model = scale_model if scale_model is not None else ScaleModel()
        self.__lambda_model_min = (
            lambda_model_min
            if lambda_model_min is not None
            else SVRModel(lambda_type="min")
        )
        self.__lambda_model_max = (
            lambda_model_max
            if lambda_model_max is not None
            else SVRModel(lambda_type="max")
        )

    def get_ans(
        self, phase_qubit_num: int | None = None, epsilon: float = 1 / 16
    ) -> list:
        """
        Runs the SAPO solver to get the optimal portfolio configuration.

        Args:
            phase_qubit_num (int, optional): The number of phase qubits to use.
                Defaults to None.
            epsilon (float, optional): The precision parameter for the optimization.
                Defaults to 1/16.

        Returns:
            The result of the SAPO solver's get_ans method.
        """
        sapo_solver = SAPO(
            data_loader=self.__data_loader,
            scale_model=self.__scale_model,
            lambda_model_min=self.__lambda_model_min,
            lambda_model_max=self.__lambda_model_max,
        )
        return sapo_solver.get_ans(phase_qubit_num, epsilon)


if __name__ == "__main__":

    def test_get_ans():
        set_random()
        STOCK_NAME = "amex"
        print(f"Open file")
        file_loader = LocalFileLoader(
            src="resources/amex_data.csv",
            read_func=lambda path: pd.read_csv(
                path, header=[0, 1], index_col=0, encoding="ascii"
            ),
        )
        num_pairs = 2
        num_assets = 3
        print(f"Load data")
        data_loader = StockDataLoader(
            data_src=file_loader,
            valid_tickers_path="resources/valid_amex_tickers_v2.json",
            num_pairs=num_pairs,
            num_assets=num_assets,
        )
        print(f"Get scale")
        scale_model = ScaleModel(mode="fixe", stock=STOCK_NAME)
        print(f"Lambda model")
        svr_matrix_scale = {"min": 4000, "max": 400}
        for typ in ["min", "max"]:
            lambda_model_filename = (
                f"resources/model/svr/{STOCK_NAME}_{num_assets}_{typ}.pkl"
            )
            lambda_model = SVRModel(
                mode="load",
                data_loader=data_loader,
                scale_model=scale_model,
                model_filename=lambda_model_filename,
                lambda_type=typ,
                matrix_scale=svr_matrix_scale[typ],
            )
            if typ == "min":
                lambda_model_min = lambda_model
            elif typ == "max":
                lambda_model_max = lambda_model
        print(f"Start solve")
        solver = PortfolioOptimization(
            data_loader=data_loader,
            scale_model=scale_model,
            lambda_model_min=lambda_model_min,
            lambda_model_max=lambda_model_max,
        )
        res = solver.get_ans(phase_qubit_num=5)

        print(f"res = {res}")

    test_get_ans()
