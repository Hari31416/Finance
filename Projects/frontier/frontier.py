import numpy as np
from pystock.portfolio import Portfolio
from pystock.models import Model
import plotly.express as px
import pandas as pd
import warnings

pd.options.mode.chained_assignment = None


class CurveFitting:
    """
    A class to implement mth order polynomial regression using the least squares method.

    Use the `fit` method to fit the model. Then predict the Y values given X values using\\
    the `predict` method.

    """

    def __init__(self) -> None:
        self.beta = None
        self.stats = None

    def fit(self, X, Y, order=3):
        """
        Polynomial regression of order m using least squares method.

        Parameters
        ----------
        X : array_like
            Independent variable.
        Y : array_like
            Dependent variable.
        order : int, optional
            Order of the polynomial. Default is 3.

        Returns
        -------
        beta : array_like
            Coefficients of the polynomial regression model.
        """
        self.n = len(X)
        Xis = np.zeros(2 * order + 1)
        Yis = np.zeros(order + 1)
        for i in range(0, 2 * order + 1):
            if i == 0:
                Xis[i] = self.n
                continue
            xi = np.sum(X**i)
            Xis[i] = xi

        for i in range(1, order + 2):
            yi = np.sum(Y * (X ** (i - 1)))
            Yis[i - 1] = yi
        A = np.zeros((order + 1, order + 1))
        for i in range(0, order + 1):
            A[i] = Xis[i : i + order + 1]
        beta = np.linalg.solve(A, Yis)
        self.beta = beta
        return beta

    def predict(self, X_l):
        """
        Predict the Y values given X values.

        Parameters
        ----------
        X_l : array_like
            Independent variable.

        Returns
        -------
        Y_l : array_like
            Predicted Y values.
        """
        Y_l = np.zeros(len(X_l))
        for i in range(0, len(self.beta)):
            Y_l += self.beta[i] * X_l**i
        return Y_l


class EfficientFrontier:
    """
    A class to plot an Efficient Frontier.

    Parameters
    ----------
    portfolio : Portfolio
        Portfolio object.
    frequency : str, optional
        Frequency of the returns. Default is "M" (monthly).
    risk_free_rate : float, optional
        The risk free rate

    Attributes
    ----------
    portfolio : Portfolio
        Portfolio object.
    frequency : str
        Frequency of the returns.
    risk_free_rate : float
        The risk free rate

    Methods
    -------
    plot_frontier(short=False, model="capm")
        Plots the Efficient Frontier.

    """

    def __init__(
        self, portfolio: Portfolio, frequency="M", risk_free_rate=1 / 3
    ) -> None:
        self.portfolio = portfolio
        self.frequency = frequency
        self.risk_free_rate = risk_free_rate
        self.__prepare()

    def __create_weight_simple(self):
        required_size = 8000
        weights = np.random.uniform(
            low=0.0, high=1.0, size=(required_size, len(self.portfolio) - 1)
        )
        weights /= np.sum(weights, axis=1)[:, np.newaxis]
        return weights

    def __create_weight_short(self):
        size = 20000
        weights = np.random.uniform(
            low=-1.0, high=1.0, size=(size, len(self.portfolio) - 1)
        )
        weights /= np.sum(weights, axis=1)[:, np.newaxis]
        weights.clip(min=-1, max=1, out=weights)
        weights = weights[weights.sum(axis=1) == 1]
        return weights

    def __create_weights(self, short=False):
        required_size = 8000
        if short:
            weights = self.__create_weight_short()
        else:
            weights = self.__create_weight_simple()
        if len(weights) < required_size:
            warnings.warn(
                f"Only {len(weights)} random points were generated. You might expect fewer points."
            )
        else:
            weights = weights[:required_size]
        return weights

    def __prepare(self):
        """Prepares the portfolio"""
        self.model = Model(self.frequency, self.risk_free_rate)
        self.model.add_portfolio(self.portfolio, weights="equal")

    def __return(self, weights, model="capm"):
        returns = np.zeros(len(weights))
        volatilities = np.zeros(len(weights))
        sharpe_ratios = np.zeros(len(weights))
        for i in range(len(weights)):
            return_, variance, std = self.model.portfolio_info(weights[i], model=model)
            returns[i] = return_
            volatilities[i] = variance
            sharpe_ratios[i] = (return_ - self.risk_free_rate) / std

        data = pd.DataFrame(
            {
                "Volatility": volatilities,
                "Return": returns,
                "Sharpe ratio": sharpe_ratios,
            }
        )
        data["Index"] = data.index
        return data

    def __remove_outliers(self, data, column):
        data = data.copy()
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        data = data[(data[column] >= Q1 - 1.5 * IQR) & (data[column] <= Q3 + 1.5 * IQR)]
        return data

    def __data_preprocess(self, data):
        data["Return2"] = data["Return"].apply(lambda x: np.round(x, 3))
        return_unique = data["Return2"].unique()
        min_volatility = []
        index = []
        for i in return_unique:
            min_volatility.append(data[data["Return2"] == i]["Volatility"].min())
            index.append(data[data["Return2"] == i]["Volatility"].idxmin())
        min_volatility = np.array(min_volatility)
        index = np.array(index)
        data_new = data.iloc[index]

        data_new.sort_values(by="Return", inplace=True)
        data_new["Return_Change"] = data_new["Return"].diff(1).fillna(0)
        data_final = self.__remove_outliers(data_new, "Return_Change")
        return data_final, index

    def __fit(self, X, Y, order=3):
        cf = CurveFitting()
        _ = cf.fit(X=X, Y=Y, order=order)
        Y_pred = cf.predict(X)

        return Y_pred

    def plot_frontier(self, short=False, model="capm"):
        """
        Creates a plot of the efficient frontier.

        Parameters
        ----------
        short : bool, optional
            Whether to allow shorting, by default False
        model : str, optional
            The model to use, by default "capm". Supported models are 'capm', 'sim'
            (If you want to use 'fff3' or 'fff5', first load the fff parameters.)

        Returns
        -------
        fig : plotly.graph_objects.Figure

        Examples
        --------
        >>> ef = EfficientFrontier(portfolio)
        >>> ef.plot_frontier()

        Notes
        -----
        You have to create a Portoflio object first. Then you need to load data. Only then you can call `plot_frontier()`.\\
        If you want to use 'fff3' or 'fff5', first load the fff parameters.
        """
        # if len(self.portfolio) == 3:
        #     self.model.portfolio_frontier(model=model)
        #     return

        weights = self.__create_weights(short=short)
        data = self.__return(weights=weights, model=model)
        data_final, _ = self.__data_preprocess(data=data)

        Y = data_final["Volatility"]
        X = data_final["Return"]

        Y_pred = self.__fit(X, Y, 3)

        data_final["Final_Return"] = X
        data_final["Final_Volatility"] = Y_pred
        weights_final = weights[data_final.index]
        customdata = weights_final.T * 100
        sharpe_ratios = data_final["Sharpe ratio"].astype(float).values

        tempelate = "Volatility deviation: %{x:.4f}<br>Expected return: %{y:.4f}"
        for i, name in enumerate(self.portfolio.stock_names):
            tempelate += f"<br>{name}: %{{customdata{[i]}:.4f}}"

        fig = px.scatter(
            x=data_final["Final_Volatility"],
            y=data_final["Final_Return"],
            labels={"x": "Volatility", "y": "Return"},
            custom_data=customdata,
            color=sharpe_ratios,
            color_continuous_scale="Viridis",
        )
        fig.update_traces(
            hovertemplate=tempelate,
            marker=dict(
                size=5,
                color=sharpe_ratios,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Sharpe ratio"),
            ),
        )
        # Update the color bar name
        fig.update_coloraxes(colorbar_title_text="Sharpe ratio")

        fig.show()
        return fig
