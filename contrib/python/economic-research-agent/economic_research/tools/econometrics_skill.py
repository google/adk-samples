# Copyright 2025 Google LLC. This software is provided as-is, without warranty or representation.
"""ADK Skill: Econometrics Sandbox. Runs regressions, correlations, and ADF tests."""

import json
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


def run_econometric_regression(
    dependent_values: list[float],
    independent_values: list[float] | list[list[float]],
    independent_names: list[str] = None,
    analysis_type: str = "OLS"
) -> str:
    """
    Runs econometric analysis (OLS, Correlation Matrix, or ADF Stationarity Test) on numerical data.
    
    Args:
        dependent_values: List of numerical values for the dependent variable (Y).
        independent_values: List of numerical values (or list of lists) for the independent variables (X).
        independent_names: Optional labels for the independent variables.
        analysis_type: The type of analysis. Must be one of: "OLS", "correlation", "ADF".
        
    Returns:
        JSON string containing the statistical results and clean tabular outputs.
    """
    try:
        # Standardize independent_values to list of lists
        if not independent_values:
            return json.dumps({"ERROR": "independent_values is empty."}, indent=2)
            
        if not isinstance(independent_values[0], list):
            # Convert single list [1.0, 2.0, ...] to [[1.0, 2.0, ...]]
            x_matrix = [independent_values]
        else:
            x_matrix = independent_values

        # Validation checks
        y_len = len(dependent_values)
        for idx, x_list in enumerate(x_matrix):
            if len(x_list) != y_len:
                return json.dumps({
                    "ERROR": f"Dimension mismatch: dependent has length {y_len}, but independent variable {idx} has length {len(x_list)}."
                }, indent=2)

        analysis_clean = analysis_type.upper().strip()

        if analysis_clean == "OLS":
            # Build DataFrame
            df = pd.DataFrame({"Y": dependent_values})
            
            names = []
            if independent_names:
                names = independent_names[:len(x_matrix)]
            # Fill missing names
            while len(names) < len(x_matrix):
                names.append(f"X{len(names) + 1}")
                
            for i, x_list in enumerate(x_matrix):
                df[names[i]] = x_list

            X = df[names]
            X = sm.add_constant(X)
            y = df["Y"]

            model = sm.OLS(y, X).fit()

            # Compile coefficients summary
            coef_summary = []
            for var in X.columns:
                coef_summary.append({
                    "Variable": var,
                    "Coefficient": f"{model.params[var]:.4f}",
                    "Std_Err": f"{model.bse[var]:.4f}",
                    "t_Stat": f"{model.tvalues[var]:.4f}",
                    "p_Value": f"{model.pvalues[var]:.4f}"
                })

            return json.dumps({
                "Analysis": "Ordinary Least Squares (OLS) Regression",
                "Observations": model.nobs,
                "R_Squared": f"{model.rsquared:.4f}",
                "Adj_R_Squared": f"{model.rsquared_adj:.4f}",
                "F_Statistic": f"{model.fvalue:.4f}",
                "Prob_F_Statistic": f"{model.f_pvalue:.4e}",
                "Coefficients": coef_summary,
                "Interpretation": (
                    "The model explains approximately "
                    f"{model.rsquared*100:.1f}% of the variance in the dependent variable. "
                    "A variable is statistically significant if its p-value is less than 0.05."
                ),
                "Source": "Econometric Sandbox Engine"
            }, indent=2)

        elif analysis_clean == "CORRELATION":
            df = pd.DataFrame({"Y": dependent_values})
            names = ["Y"]
            if independent_names:
                names.extend(independent_names[:len(x_matrix)])
            while len(names) - 1 < len(x_matrix):
                names.append(f"X{len(names)}")
                
            for i, x_list in enumerate(x_matrix):
                df[names[i+1]] = x_list

            corr_matrix = df.corr(method="pearson").round(4).to_dict()
            return json.dumps({
                "Analysis": "Pearson Correlation Matrix",
                "Matrix": corr_matrix,
                "Interpretation": "Values close to 1 or -1 indicate strong linear correlation. Values near 0 indicate no linear correlation.",
                "Source": "Econometric Sandbox Engine"
            }, indent=2)

        elif analysis_clean == "ADF":
            # Runs ADF on dependent_values as a single timeseries
            result = adfuller(dependent_values)
            critical_values = {k: f"{v:.4f}" for k, v in result[4].items()}
            
            return json.dumps({
                "Analysis": "Augmented Dickey-Fuller (ADF) Stationarity Test",
                "ADF_Statistic": f"{result[0]:.4f}",
                "p_Value": f"{result[1]:.4f}",
                "Lags_Used": result[2],
                "Observations": result[3],
                "Critical_Values": critical_values,
                "Is_Stationary": "Yes" if result[1] < 0.05 else "No",
                "Interpretation": (
                    "If the p-value is less than 0.05, we reject the null hypothesis of a unit root, "
                    "indicating the timeseries is stationary."
                ),
                "Source": "Econometric Sandbox Engine"
            }, indent=2)

        else:
            return json.dumps({
                "ERROR": f"Unsupported analysis_type: '{analysis_type}'. Must be OLS, correlation, or ADF."
            }, indent=2)

    except Exception as e:
        return json.dumps({"ERROR": f"Econometric computation failed: {str(e)}"}, indent=2)
