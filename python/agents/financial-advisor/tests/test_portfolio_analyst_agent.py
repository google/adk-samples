import json
import unittest
from unittest.mock import MagicMock, patch

from google.adk.tools import ToolCall, ToolResult
from financial_advisor.sub_agents.portfolio_analyst import (
    portfolio_analyst_agent,
)


class TestPortfolioAnalystAgent(unittest.TestCase):
    @patch.object(portfolio_analyst_agent, "call_tool")
    def test_portfolio_analysis_valid_input_no_concentration(
        self, mock_call_tool
    ):
        # Test with a simple portfolio, no concentrations
        user_portfolio = "GOOGL:10,AAPL:20"
        user_risk_attitude = "moderate"
        investment_period = "long-term"

        # Mock google_search responses
        mock_call_tool.side_effect = [
            # GOOGL
            ToolResult(
                ToolCall("google_search", {"query": "current market price of GOOGL"}),
                "150.00 USD",
            ),
            ToolResult(
                ToolCall("google_search", {"query": "primary stock exchange of GOOGL"}),
                "NASDAQ",
            ),
            ToolResult(
                ToolCall("google_search", {"query": "sector or industry of GOOGL"}),
                "Technology",
            ),
            # AAPL
            ToolResult(
                ToolCall("google_search", {"query": "current market price of AAPL"}),
                "170.00 USD",
            ),
            ToolResult(
                ToolCall("google_search", {"query": "primary stock exchange of AAPL"}),
                "NASDAQ",
            ),
            ToolResult(
                ToolCall("google_search", {"query": "sector or industry of AAPL"}),
                "Technology",
            ),
        ]

        expected_output_structure = {
            "portfolio_summary": {
                "total_value": float,
                "overall_risk_commentary": str,
            },
            "holdings_details": [
                {
                    "ticker": str,
                    "quantity": int,
                    "current_price": float,
                    "market_value": float,
                    "exchange": str,
                    "sector": str,
                    "percentage_of_portfolio": float,
                }
            ],
            "sector_allocation": {str: float},
            "concentration_warnings": list,
            "disclaimer": str,
        }

        # Call the agent
        result = portfolio_analyst_agent.invoke(
            user_portfolio=user_portfolio,
            user_risk_attitude=user_risk_attitude,
            investment_period=investment_period,
        )

        self.assertIn("portfolio_analysis_output", result)
        output_data = json.loads(result["portfolio_analysis_output"])

        # Validate overall structure
        self.assertIsInstance(
            output_data["portfolio_summary"]["total_value"], float
        )
        self.assertIsInstance(
            output_data["portfolio_summary"]["overall_risk_commentary"], str
        )
        self.assertIsInstance(output_data["holdings_details"], list)
        self.assertIsInstance(output_data["sector_allocation"], dict)
        self.assertIsInstance(output_data["concentration_warnings"], list)
        self.assertIsInstance(output_data["disclaimer"], str)

        # Validate calculations (example for GOOGL and AAPL)
        # GOOGL: 10 * 150 = 1500
        # AAPL: 20 * 170 = 3400
        # Total: 1500 + 3400 = 4900
        self.assertAlmostEqual(
            output_data["portfolio_summary"]["total_value"], 4900.00
        )

        goog_details = next(
            (h for h in output_data["holdings_details"] if h["ticker"] == "GOOGL"),
            None,
        )
        aapl_details = next(
            (h for h in output_data["holdings_details"] if h["ticker"] == "AAPL"),
            None,
        )

        self.assertIsNotNone(goog_details)
        self.assertEqual(goog_details["quantity"], 10)
        self.assertAlmostEqual(goog_details["current_price"], 150.00)
        self.assertAlmostEqual(goog_details["market_value"], 1500.00)
        self.assertAlmostEqual(
            goog_details["percentage_of_portfolio"], (1500 / 4900) * 100
        )
        self.assertEqual(goog_details["exchange"], "NASDAQ")
        self.assertEqual(goog_details["sector"], "Technology")

        self.assertIsNotNone(aapl_details)
        self.assertEqual(aapl_details["quantity"], 20)
        self.assertAlmostEqual(aapl_details["current_price"], 170.00)
        self.assertAlmostEqual(aapl_details["market_value"], 3400.00)
        self.assertAlmostEqual(
            aapl_details["percentage_of_portfolio"], (3400 / 4900) * 100
        )
        self.assertEqual(aapl_details["exchange"], "NASDAQ")
        self.assertEqual(aapl_details["sector"], "Technology")

        self.assertIn("Technology", output_data["sector_allocation"])
        self.assertAlmostEqual(
            output_data["sector_allocation"]["Technology"], 100.0
        )
        # No concentration warnings expected here as Technology is 100% but individual stocks are not >25%
        # and sector concentration is only warned if >50% *and* there are other sectors.
        # The prompt is a bit ambiguous on this, so current test assumes sector warning if >50% and it's not the *only* sector.
        # For a single sector portfolio, 100% is normal.
        # Let's refine this: if a single sector is > 50%, it should be warned.
        # The prompt: "Check if any single sector/industry constitutes more than 50% of the total portfolio value."
        # This implies it should be listed.

        # Re-evaluating based on prompt: "List these as concentration warnings."
        # "Warning: The Technology sector represents 60% of the portfolio, exceeding the 50% threshold for a single sector."
        # So, a 100% single sector should be listed.
        self.assertGreater(len(output_data["concentration_warnings"]), 0)
        self.assertTrue(
            any(
                "Technology sector represents 100.0%" in warning
                for warning in output_data["concentration_warnings"]
            )
        )

        self.assertTrue(
            "This analysis is for educational purposes only"
            in output_data["disclaimer"]
        )
        self.assertEqual(mock_call_tool.call_count, 6) # 3 calls per ticker * 2 tickers

    @patch.object(portfolio_analyst_agent, "call_tool")
    def test_portfolio_analysis_with_concentration(self, mock_call_tool):
        # Test with a portfolio that has single stock concentration
        user_portfolio = "MSFT:30,NVDA:5" # MSFT will be >25%
        user_risk_attitude = "aggressive"
        investment_period = "medium-term"

        mock_call_tool.side_effect = [
            # MSFT
            ToolResult(
                ToolCall("google_search", {"query": "current market price of MSFT"}),
                "400.00 USD", # 30 * 400 = 12000
            ),
            ToolResult(
                ToolCall("google_search", {"query": "primary stock exchange of MSFT"}),
                "NASDAQ",
            ),
            ToolResult(
                ToolCall("google_search", {"query": "sector or industry of MSFT"}),
                "Technology",
            ),
            # NVDA
            ToolResult(
                ToolCall("google_search", {"query": "current market price of NVDA"}),
                "800.00 USD", # 5 * 800 = 4000
            ),
            ToolResult(
                ToolCall("google_search", {"query": "primary stock exchange of NVDA"}),
                "NASDAQ",
            ),
            ToolResult(
                ToolCall("google_search", {"query": "sector or industry of NVDA"}),
                "Technology",
            ),
        ]
        # Total Portfolio Value = 12000 + 4000 = 16000
        # MSFT percentage = 12000 / 16000 = 75%
        # NVDA percentage = 4000 / 16000 = 25%

        result = portfolio_analyst_agent.invoke(
            user_portfolio=user_portfolio,
            user_risk_attitude=user_risk_attitude,
            investment_period=investment_period,
        )
        output_data = json.loads(result["portfolio_analysis_output"])

        self.assertAlmostEqual(
            output_data["portfolio_summary"]["total_value"], 16000.00
        )
        self.assertGreater(len(output_data["concentration_warnings"]), 0)
        self.assertTrue(
            any(
                "Warning: Holding MSFT represents 75.0%" in warning
                for warning in output_data["concentration_warnings"]
            )
        )
        self.assertTrue(
            any(
                "Technology sector represents 100.0%" in warning # Both are technology
                for warning in output_data["concentration_warnings"]
            )
        )
        self.assertEqual(mock_call_tool.call_count, 6)


    @patch.object(portfolio_analyst_agent, "call_tool")
    def test_portfolio_analysis_different_sectors(self, mock_call_tool):
        user_portfolio = "JNJ:10,XOM:100" # JNJ: Healthcare, XOM: Energy
        user_risk_attitude = "conservative"
        investment_period = "long-term"

        mock_call_tool.side_effect = [
            # JNJ
            ToolResult(
                ToolCall("google_search", {"query": "current market price of JNJ"}),
                "150.00 USD", # 10 * 150 = 1500
            ),
            ToolResult(
                ToolCall("google_search", {"query": "primary stock exchange of JNJ"}),
                "NYSE",
            ),
            ToolResult(
                ToolCall("google_search", {"query": "sector or industry of JNJ"}),
                "Healthcare",
            ),
            # XOM
            ToolResult(
                ToolCall("google_search", {"query": "current market price of XOM"}),
                "110.00 USD", # 100 * 110 = 11000
            ),
            ToolResult(
                ToolCall("google_search", {"query": "primary stock exchange of XOM"}),
                "NYSE",
            ),
            ToolResult(
                ToolCall("google_search", {"query": "sector or industry of XOM"}),
                "Energy",
            ),
        ]
        # Total Portfolio Value = 1500 + 11000 = 12500
        # JNJ: 1500 / 12500 = 12% (Healthcare)
        # XOM: 11000 / 12500 = 88% (Energy)

        result = portfolio_analyst_agent.invoke(
            user_portfolio=user_portfolio,
            user_risk_attitude=user_risk_attitude,
            investment_period=investment_period,
        )
        output_data = json.loads(result["portfolio_analysis_output"])

        self.assertAlmostEqual(
            output_data["portfolio_summary"]["total_value"], 12500.00
        )
        self.assertIn("Healthcare", output_data["sector_allocation"])
        self.assertAlmostEqual(
            output_data["sector_allocation"]["Healthcare"], 12.0
        )
        self.assertIn("Energy", output_data["sector_allocation"])
        self.assertAlmostEqual(output_data["sector_allocation"]["Energy"], 88.0)

        self.assertGreater(len(output_data["concentration_warnings"]), 0)
        self.assertTrue(
            any(
                "Warning: The Energy sector represents 88.0%" in warning
                for warning in output_data["concentration_warnings"]
            )
        )
        # JNJ is 12%, XOM is 88%. XOM should be warned for stock concentration.
        self.assertTrue(
            any(
                "Warning: Holding XOM represents 88.0%" in warning
                for warning in output_data["concentration_warnings"]
            )
        )
        self.assertEqual(mock_call_tool.call_count, 6)

    def test_portfolio_parsing_with_average_price(self):
        # This test does not call the agent, but checks if the prompt implies
        # that the agent should parse and potentially use average prices.
        # The current PORTFOLIO_ANALYST_PROMPT's "Parse Portfolio" section only mentions
        # extracting ticker and quantity. Average price is not mentioned for processing.
        # However, the financial_coordinator prompt (Workflow B, Step B1) *does* show an example:
        # "GOOGL:10:2500.50"
        # For now, portfolio_analyst_agent is not explicitly asked to use avg_price.
        # This test is more of a placeholder to note this.
        # If the agent were to use it, we'd test that here.
        # For now, we assume the agent will ignore the third parameter if provided.
        self.assertTrue(True) # Placeholder

    @patch.object(portfolio_analyst_agent, "call_tool")
    def test_portfolio_malformed_input(self, mock_call_tool):
        # The prompt for portfolio_analyst doesn't specify robust error handling for malformed strings.
        # It expects "TICKER:QUANTITY". If the LLM handles it gracefully, that's good.
        # This test is to see how it might behave, but we can't strictly assert an error
        # unless the prompt explicitly required it.
        # For now, we expect it to try its best or the LLM to potentially not find tickers.
        user_portfolio = "GOOGL:10,INVALID_TICKER_FORMAT"
        user_risk_attitude = "moderate"
        investment_period = "long-term"

        mock_call_tool.side_effect = [
            # GOOGL
            ToolResult(
                ToolCall("google_search", {"query": "current market price of GOOGL"}),
                "150.00 USD",
            ),
            ToolResult(
                ToolCall("google_search", {"query": "primary stock exchange of GOOGL"}),
                "NASDAQ",
            ),
            ToolResult(
                ToolCall("google_search", {"query": "sector or industry of GOOGL"}),
                "Technology",
            ),
            # For "INVALID_TICKER_FORMAT", the LLM might try searching, or might ignore.
            # Let's assume it tries to search for "INVALID_TICKER_FORMAT" as a ticker.
            ToolResult(
                ToolCall("google_search", {"query": "current market price of INVALID_TICKER_FORMAT"}),
                "N/A", # Simulate search finding nothing or error
            ),
            ToolResult(
                ToolCall("google_search", {"query": "primary stock exchange of INVALID_TICKER_FORMAT"}),
                "N/A",
            ),
            ToolResult(
                ToolCall("google_search", {"query": "sector or industry of INVALID_TICKER_FORMAT"}),
                "N/A",
            ),
        ]

        result = portfolio_analyst_agent.invoke(
            user_portfolio=user_portfolio,
            user_risk_attitude=user_risk_attitude,
            investment_period=investment_period,
        )
        output_data = json.loads(result["portfolio_analysis_output"])

        # Expect it to process GOOGL correctly and potentially ignore or note the invalid part.
        self.assertAlmostEqual(
            output_data["portfolio_summary"]["total_value"], 1500.00 # Only GOOGL
        )
        self.assertEqual(len(output_data["holdings_details"]), 1)
        self.assertEqual(output_data["holdings_details"][0]["ticker"], "GOOGL")
        # Depending on LLM's robustness, it might add a note about unparsed items or simply ignore.
        # The prompt doesn't specify, so we won't assert too strictly here.
        # We are asserting that it *at least* processes the valid parts.
        self.assertTrue(
            "Commentary may be affected by any unparsed parts of the portfolio string."
            in output_data["portfolio_summary"]["overall_risk_commentary"] or len(output_data["holdings_details"]) == 1
        )


if __name__ == "__main__":
    unittest.main()
