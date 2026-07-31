# Copyright 2025 Google LLC. This software is provided as-is, without warranty or representation.
"""ADK Skill: Commercial Real Estate (CRE) Underwriting Tool."""

import json


def underwrite_deal_leverage(
    purchase_price: float,
    rent_monthly: float,
    down_payment_pct: float = 20.0,
    interest_rate: float = 6.5,
    operating_expenses_pct: float = 40.0,
    vacancy_rate: float = 5.0,
    amortization_years: int = 30
) -> str:
    """
    Performs standard Commercial Real Estate (CRE) leveraged yield underwriting.
    
    Args:
        purchase_price: Total purchase price of the property (USD).
        rent_monthly: Current total monthly rent collected across all units (USD).
        down_payment_pct: Down payment percentage (e.g. 20.0 for 20%).
        interest_rate: Annual mortgage interest rate percentage (e.g. 6.5 for 6.5%).
        operating_expenses_pct: Operating expense ratio percentage (e.g. 40.0 for 40%).
        vacancy_rate: Expected vacancy rate percentage (e.g. 5.0 for 5%).
        amortization_years: Mortgage amortization period in years (default is 30).
        
    Returns:
        JSON string containing pro-forma underwriting cash flow table and key metrics.
    """
    try:
        # Convert percentages to decimals
        dp_dec = down_payment_pct / 100.0
        ir_dec = interest_rate / 100.0
        oe_dec = operating_expenses_pct / 100.0
        vac_dec = vacancy_rate / 100.0

        # Calculations
        loan_amount = purchase_price * (1.0 - dp_dec)
        equity_required = purchase_price * dp_dec

        gross_potential_rent = rent_monthly * 12.0
        vacancy_loss = gross_potential_rent * vac_dec
        effective_gross_income = gross_potential_rent - vacancy_loss
        operating_expenses = effective_gross_income * oe_dec
        noi = effective_gross_income - operating_expenses

        # Monthly Debt Service calculation (amortizing)
        if ir_dec > 0:
            monthly_rate = ir_dec / 12.0
            num_months = amortization_years * 12
            monthly_payment = loan_amount * (monthly_rate * (1.0 + monthly_rate) ** num_months) / (((1.0 + monthly_rate) ** num_months) - 1.0)
            annual_debt_service = monthly_payment * 12.0
        else:
            monthly_payment = 0.0
            annual_debt_service = 0.0

        cash_flow = noi - annual_debt_service

        # Financial Metrics
        cap_rate = (noi / purchase_price) * 100.0 if purchase_price else 0.0
        cash_on_cash = (cash_flow / equity_required) * 100.0 if equity_required else 0.0
        dscr = noi / annual_debt_service if annual_debt_service else float("inf")
        debt_yield = (noi / loan_amount) * 100.0 if loan_amount else 0.0

        pro_forma_table = [
            {"Line Item": "Gross Potential Rent (Annual)", "Value": f"${gross_potential_rent:,.2f}"},
            {"Line Item": f"Vacancy Loss ({vacancy_rate:.1f}%)", "Value": f"-${vacancy_loss:,.2f}"},
            {"Line Item": "Effective Gross Income (EGI)", "Value": f"${effective_gross_income:,.2f}"},
            {"Line Item": f"Operating Expenses ({operating_expenses_pct:.1f}%)", "Value": f"-${operating_expenses:,.2f}"},
            {"Line Item": "Net Operating Income (NOI)", "Value": f"${noi:,.2f}"},
            {"Line Item": "Annual Debt Service", "Value": f"-${annual_debt_service:,.2f}"},
            {"Line Item": "Net Cash Flow (Leveraged)", "Value": f"${cash_flow:,.2f}"}
        ]

        return json.dumps({
            "Acquisition_Summary": {
                "Purchase_Price": f"${purchase_price:,.2f}",
                "Equity_Required": f"${equity_required:,.2f}",
                "Loan_Amount": f"${loan_amount:,.2f}",
                "Monthly_Mortgage_Payment": f"${monthly_payment:,.2f}"
            },
            "Pro_Forma_Cash_Flow": pro_forma_table,
            "Performance_Metrics": {
                "Unleveraged_Cap_Rate": f"{cap_rate:.2f}%",
                "Leveraged_Cash_on_Cash_Return": f"{cash_on_cash:.2f}%",
                "Debt_Service_Coverage_Ratio_DSCR": f"{dscr:.2f}" if dscr != float("inf") else "N/A",
                "Debt_Yield": f"{debt_yield:.2f}%"
            },
            "Underwriting_Assumptions": {
                "Down_Payment_Pct": f"{down_payment_pct:.1f}%",
                "Mortgage_Interest_Rate": f"{interest_rate:.1f}%",
                "Amortization_Period": f"{amortization_years} Years"
            },
            "Source": "Investment Acquisitions Underwriting Tool"
        }, indent=2)

    except Exception as e:
        return json.dumps({"ERROR": f"Underwriting calculation failed: {str(e)}"}, indent=2)
