import numpy as np
import pandas as pd
from arch.univariate import ARX, GARCH, StudentsT

# Fit our AR-GARCH model with Student-t errors to the returns data.
def fitGARCHModel(logReturns):
    model = ARX(logReturns, lags=0, rescale=True)
    model.volatility = GARCH(1, 0, 1)
    model.distribution = StudentsT()
    results = model.fit(disp='off')
    return results


# Extracting parameters of interest from our fitted model.
def extractParameters(results):
    parameters = results.params
    intercept = float(parameters["Const"])
    phi = 0.0
    omega = float(parameters["omega"])
    alpha = float(parameters["alpha[1]"])
    beta = float(parameters["beta[1]"])
    nu = float(parameters["nu"])
    return intercept, phi, omega, alpha, beta, nu


# 100 day Monte-Carlo simulation of future returns based on the fitted model.
def simulatePaths(logReturns, results, simulations=1000, days=100, seed=42):
    intercept, phi, omega, alpha, beta, nu = extractParameters(results)

    # Information from the last observed return and volatility to start the simulation.
    lastReturnPCT = float(logReturns.iloc[-1]) * 100
    lastEpsilon = float(results.resid.iloc[-1])
    lastSigma = float(results.conditional_volatility.iloc[-1])

    returnLag = np.full(simulations, lastReturnPCT)
    epsilonLag = np.full(simulations, lastEpsilon)
    sigmaLag = np.full(simulations, lastSigma)

    # Define random shocks following a student-t distribution with unit variance.
    rng = np.random.default_rng(seed)
    if nu > 2:
        z = rng.standard_t(df=nu, size=(days, simulations)) / np.sqrt(nu/(nu-2))
    else:
        z = rng.standard_normal((days, simulations))

    # Define an array to hold information at each time step.
    simulationMatrix = np.empty((days, simulations))

    for i in range(days):
        nextSigma = np.sqrt(omega + alpha * epsilonLag**2 + beta * sigmaLag**2)
        nextMu = intercept + phi * returnLag
        nextEpsilon = nextSigma * z[i, :]
        nextReturn = nextMu + nextEpsilon

        # Store the simulated returns for this time step.
        simulationMatrix[i, :] = nextReturn

        # Update the lagged values for the next iteration.
        returnLag, epsilonLag, sigmaLag = nextReturn, nextEpsilon, nextSigma

    simulationMatrix /= 100.0

    dfSimulations = pd.DataFrame(simulationMatrix, columns=[f'Simulation_{i+1}' for i in range(simulations)])

    return dfSimulations


def calculateRiskMetrics(simulatedReturns, lastClose, alpha=0.95):
    # Convert returns back to price paths for one day horizon.
    oneDayPricePaths = np.exp(simulatedReturns.iloc[0, :]) - 1.0
    losses = -oneDayPricePaths

    # Calculate our simulated price paths. Return first 100 for visualization.
    priceHorizon = lastClose * np.exp(np.cumsum(simulatedReturns.to_numpy(), axis=0))
    dfPriceHorizon = pd.DataFrame(priceHorizon[:, :100]).reset_index().melt(
        id_vars='index', var_name='Simulation', value_name='Price'
    )
    dfPriceHorizon.rename(columns={'index': 'Day'}, inplace=True)

    # Calculate Value at Risk (VaR) and Expected Shortfall (CVaR).
    VaR = np.quantile(losses, alpha)
    CVaR = losses[losses >= VaR].mean() if np.any(losses >= VaR) else VaR

    return oneDayPricePaths, dfPriceHorizon, VaR, CVaR



