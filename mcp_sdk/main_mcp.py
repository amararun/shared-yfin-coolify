from typing import List
import yfinance as yf
import pandas as pd
from datetime import datetime
import time
import traceback
import numpy as np
from mcp.server.fastmcp import FastMCP
import os
from starlette.applications import Starlette
from starlette.routing import Mount

# Create MCP server with explicit configuration
mcp = FastMCP(
    name="YFIN Financial Data",
    description="Financial data server using yfinance",
    version="1.0.0"
)

# Define helper functions for data fetching
def fetch_balance_sheet_as_text(tickers: List[str]):
    try:
        all_text = ""
        for ticker in tickers:
            ticker = ticker.strip()
            tkr = yf.Ticker(ticker)
            bs_data = tkr.balance_sheet
            if bs_data is not None and not bs_data.empty:
                bs_data.reset_index(inplace=True)
                bs_data.insert(0, 'Ticker', ticker)
                text = bs_data.to_csv(index=False, sep='|')
                all_text += text + "\n"
            else:
                all_text += f"No balance sheet data available for {ticker}\n"
            time.sleep(10)
        return all_text
    except Exception as e:
        return f"Error: {str(e)}"

def fetch_income_statement_as_text(tickers: List[str]):
    try:
        all_text = ""
        for ticker in tickers:
            ticker = ticker.strip()
            tkr = yf.Ticker(ticker)
            income_stmt = tkr.income_stmt
            if income_stmt is not None and not income_stmt.empty:
                income_stmt.reset_index(inplace=True)
                income_stmt.insert(0, 'Ticker', ticker)
                text = income_stmt.to_csv(index=False, sep='|')
                all_text += text + "\n"
            else:
                all_text += f"No income statement data available for {ticker}\n"
            time.sleep(10)
        return all_text
    except Exception as e:
        return f"Error: {str(e)}"

def fetch_cash_flow_as_text(tickers: List[str]):
    try:
        all_text = ""
        for ticker in tickers:
            ticker = ticker.strip()
            tkr = yf.Ticker(ticker)
            cash_flow = tkr.cashflow
            if cash_flow is not None and not cash_flow.empty:
                cash_flow.reset_index(inplace=True)
                cash_flow.insert(0, 'Ticker', ticker)
                text = cash_flow.to_csv(index=False, sep='|')
                all_text += text + "\n"
            else:
                all_text += f"No cash flow statement data available for {ticker}\n"
            time.sleep(10)
        return all_text
    except Exception as e:
        return f"Error: {str(e)}"

def get_adj_close(tickers: List[str], start_date: str, end_date: str):
    try:
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError as e:
        return {"error": f"Invalid date format: {str(e)}"}

    try:
        all_data = pd.DataFrame()
        for ticker_symbol in tickers:
            ticker = yf.Ticker(ticker_symbol)
            ticker_data = ticker.history(start=start_date, end=end_date)
            if not ticker_data.empty:
                price_column = 'Close' if 'Adj Close' not in ticker_data.columns else 'Adj Close'
                all_data[ticker_symbol] = ticker_data[price_column]
        
        if all_data.empty:
            return {"error": "No data available for the specified parameters"}
            
        json_data = {}
        for date in all_data.index:
            date_str = date.strftime("%Y-%m-%d")
            json_data[date_str] = {}
            for ticker in tickers:
                if ticker in all_data.columns:
                    json_data[date_str][ticker] = all_data.at[date, ticker]
                else:
                    json_data[date_str][ticker] = "Data not available"
        
        return json_data
    except Exception as e:
        return {"error": f"An error occurred while fetching data: {str(e)}"}

def get_market_data(tickers: List[str]):
    market_data = {}
    try:
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                if not info:
                    market_data[ticker] = {
                        "error": "No data available",
                        "market_cap": None,
                        "free_float": None,
                        "total_shares_outstanding": None
                    }
                    continue
                
                data = {
                    "market_cap": info.get("marketCap"),
                    "free_float": info.get("floatShares"),
                    "total_shares_outstanding": info.get("sharesOutstanding")
                }
                market_data[ticker] = data
                
            except Exception as ticker_error:
                market_data[ticker] = {
                    "error": f"Failed to process: {str(ticker_error)}",
                    "market_cap": None,
                    "free_float": None,
                    "total_shares_outstanding": None
                }
                
        return market_data
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

# Define MCP tools
@mcp.tool()
def get_balance_sheet(tickers: str) -> str:
    """Get balance sheet data for one or more companies"""
    tickers_list = tickers.split(",")
    return fetch_balance_sheet_as_text(tickers_list)

@mcp.tool()
def get_income_statement(tickers: str) -> str:
    """Get income statement data for one or more companies"""
    tickers_list = tickers.split(",")
    return fetch_income_statement_as_text(tickers_list)

@mcp.tool()
def get_cash_flow(tickers: str) -> str:
    """Get cash flow statement data for one or more companies"""
    tickers_list = tickers.split(",")
    return fetch_cash_flow_as_text(tickers_list)

@mcp.tool()
def get_adj_close_prices(tickers: str, start_date: str, end_date: str) -> dict:
    """Get adjusted closing prices for one or more tickers within a date range"""
    tickers_list = tickers.split(",")
    return get_adj_close(tickers_list, start_date, end_date)

@mcp.tool()
def get_market_metrics(tickers: str) -> dict:
    """Get market metrics (market cap, float, shares) for one or more tickers"""
    tickers_list = tickers.split(",")
    return get_market_data(tickers_list)

# Create the ASGI application with SSE support
app = Starlette(routes=[
    Mount("/mcp", app=mcp.sse_app(), name="mcp")
])

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
