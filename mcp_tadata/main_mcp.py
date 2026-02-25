from fastapi import FastAPI
from typing import List
import yfinance as yf
import pandas as pd
from datetime import datetime
import time
from fastapi import FastAPI, Request, Response, Query
import traceback
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
# Import FastApiMCP for MCP server integration
from fastapi_mcp import FastApiMCP
import logging
import uuid
import sys
import os
from starlette.middleware.base import BaseHTTPMiddleware
import json
import asyncio
from collections import defaultdict
import httpx  # Import httpx for custom client configuration
from tigzig_api_monitor import APIMonitorMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("yfinance-api")

# ============================================================
# STARTUP MARKER - Version with tigzig-api-monitor v1.3
# ============================================================
print("=" * 60)
print("YFIN_COOLIFY MCP API v1.4 - HARDENED (rate limit, concurrency, CORS, error sanitization)")
print("=" * 60)

# Debug: Print env vars (masked) to verify they are loaded
def mask_env(name):
    val = os.getenv(name, "")
    if val:
        return val[:3] + "***" + val[-2:] if len(val) > 5 else "***"
    return "NOT SET"

print(f"API_MONITOR_URL: {mask_env('API_MONITOR_URL')}")
print(f"API_MONITOR_KEY: {mask_env('API_MONITOR_KEY')}")
print("=" * 60)

# ============================================================
# RATE LIMITING - configurable via env var
# ============================================================
RATE_LIMIT = os.getenv("RATE_LIMIT", "30/minute")
limiter = Limiter(key_func=get_remote_address)

# ============================================================
# CONCURRENCY LIMITS - configurable via env vars
# ============================================================
MAX_CONCURRENT_PER_IP = int(os.getenv("MAX_CONCURRENT_PER_IP", "3"))
MAX_CONCURRENT_GLOBAL = int(os.getenv("MAX_CONCURRENT_GLOBAL", "6"))
_concurrency_counts: dict = defaultdict(int)
_concurrency_global: int = 0
_concurrency_lock = asyncio.Lock()


async def check_concurrency(client_ip: str):
    """Acquire a concurrency slot or raise 429."""
    global _concurrency_global
    async with _concurrency_lock:
        if _concurrency_global >= MAX_CONCURRENT_GLOBAL:
            raise HTTPException(status_code=429, detail="Server is busy. Please try again shortly.")
        if _concurrency_counts[client_ip] >= MAX_CONCURRENT_PER_IP:
            raise HTTPException(status_code=429, detail="Too many concurrent requests. Please try again shortly.")
        _concurrency_global += 1
        _concurrency_counts[client_ip] += 1


async def release_concurrency(client_ip: str):
    """Release a concurrency slot."""
    global _concurrency_global
    async with _concurrency_lock:
        _concurrency_global = max(0, _concurrency_global - 1)
        _concurrency_counts[client_ip] = max(0, _concurrency_counts[client_ip] - 1)
        if _concurrency_counts[client_ip] == 0:
            del _concurrency_counts[client_ip]


app = FastAPI()

# Add rate limiter to app state and error handler
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

from fastapi.responses import JSONResponse
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"error": f"Rate limit exceeded. Limit: {RATE_LIMIT}."})

from fastapi import HTTPException

# Create a middleware for request logging
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Log the request
        client_host = request.client.host if request.client else "unknown"
        logger.info(f"Request [{request_id}]: {request.method} {request.url.path} from {client_host}")
        
        # Try to log query parameters if any
        if request.query_params:
            logger.info(f"Request [{request_id}] params: {dict(request.query_params)}")
        
        start_time = time.time()
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Log the response
            logger.info(f"Response [{request_id}]: {response.status_code} (took {process_time:.4f}s)")
            return response
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(f"Request [{request_id}] failed after {process_time:.4f}s: {str(e)}")
            logger.error(traceback.format_exc())
            return Response(
                content=json.dumps({"error": "Internal server error. Please try again later."}),
                status_code=500,
                media_type="application/json"
            )

# Add the logging middleware
app.add_middleware(RequestLoggingMiddleware)

# Add CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add API Monitor middleware (logs to centralized tigzig logger service)
# v1.3.0: Whitelist mode - only log OUR endpoints, ignore all scanner junk
app.add_middleware(
    APIMonitorMiddleware,
    app_name="YFIN_COOLIFY",
    include_prefixes=("/get-", "/excel/"),  # Only log our actual API endpoints
)

@app.middleware("http")
async def strip_yfin_prefix(request: Request, call_next):
    if request.url.path.startswith("/yfin"):
        request.scope["path"] = request.url.path[5:]  # Remove '/yfin' prefix
    response = await call_next(request)
    return response

# Define function to get annual balance sheet for multiple tickers
def fetch_balance_sheet_as_text(tickers: List[str]):
    try:
        # Initialize an empty string to store the concatenated text
        all_text = ""

        # Process each ticker symbol
        for ticker in tickers:
            # Remove any leading or trailing spaces from the ticker symbol
            ticker = ticker.strip()
            logger.info(f"Fetching balance sheet data for {ticker}")

            # Fetch balance sheet data for the current ticker symbol
            tkr = yf.Ticker(ticker)
            bs_data = tkr.balance_sheet

            # Check if balance sheet data is available and not empty
            if bs_data is not None and not bs_data.empty:
                # Reset index to include balance sheet line item descriptions as a regular column
                bs_data.reset_index(inplace=True)
                # Add a new column for the ticker symbol
                bs_data.insert(0, 'Ticker', ticker)
                # Convert DataFrame to text format with pipe "|" delimiter
                text = bs_data.to_csv(index=False, sep='|')
                # Concatenate the text with the all_text string
                all_text += text + "\n"
                logger.info(f"Successfully fetched balance sheet data for {ticker}")
            else:
                # If no balance sheet data is available for the current ticker, append a message to all_text
                all_text += f"No balance sheet data available for {ticker}\n"
                logger.warning(f"No balance sheet data available for {ticker}")

            # Introduce a 10-second delay before processing the next ticker symbol
            logger.debug(f"Waiting 10 seconds before processing next ticker")
            time.sleep(10)

        return all_text
    except Exception as e:
        logger.error(f"Error fetching balance sheet: {str(e)}")
        logger.error(traceback.format_exc())
        return "Error: Failed to fetch balance sheet data. Please try again later."

# Define the new function to get adjusted closing prices
def get_adj_close(tickers: List[str], start_date: str, end_date: str):
    try:
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError as e:
        return {"error": f"Invalid date format: {str(e)}"}

    try:
        print(f"Fetching data for tickers: {tickers}")
        print(f"Date range: {start_date} to {end_date}")
        
        # Initialize empty DataFrame to store results
        all_data = pd.DataFrame()
        
        # Fetch data for each ticker using Ticker().history()
        for ticker_symbol in tickers:
            print(f"Fetching data for {ticker_symbol}")
            ticker = yf.Ticker(ticker_symbol)
            ticker_data = ticker.history(start=start_date, end=end_date)
            
            print(f"Data received for {ticker_symbol}:")
            print(ticker_data.columns)
            print(f"Shape: {ticker_data.shape}")
            
            if not ticker_data.empty:
                # Use Close price if Adj Close is not available
                price_column = 'Close' if 'Adj Close' not in ticker_data.columns else 'Adj Close'
                all_data[ticker_symbol] = ticker_data[price_column]
        
        if all_data.empty:
            print("No data available for any ticker")
            return {"error": "No data available for the specified parameters"}
            
        print("\nFinal processed data:")
        print(all_data)
        
        # Convert to JSON format
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
        logger.error(f"Error fetching adj close: {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": "An error occurred while fetching price data. Please try again later."}

# Define function to fetch annual income statement for multiple tickers
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
        logger.error(f"Error fetching income statement: {str(e)}")
        logger.error(traceback.format_exc())
        return "Error: Failed to fetch income statement data. Please try again later."

# Define function to fetch cash flow statement for multiple tickers
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
        logger.error(f"Error fetching cash flow: {str(e)}")
        logger.error(traceback.format_exc())
        return "Error: Failed to fetch cash flow data. Please try again later."

# Define function to fetch quarterly income statement for multiple tickers
def fetch_quarterly_income_statement_as_text(tickers: List[str]):
    try:
        all_text = ""
        for ticker in tickers:
            ticker = ticker.strip()
            tkr = yf.Ticker(ticker)
            quarterly_income_stmt = tkr.quarterly_income_stmt
            if quarterly_income_stmt is not None and not quarterly_income_stmt.empty:
                quarterly_income_stmt.reset_index(inplace=True)
                quarterly_income_stmt.insert(0, 'Ticker', ticker)
                text = quarterly_income_stmt.to_csv(index=False, sep='|')
                all_text += text + "\n"
            else:
                all_text += f"No quarterly income statement data available for {ticker}\n"
            time.sleep(10)
        return all_text
    except Exception as e:
        logger.error(f"Error fetching quarterly income statement: {str(e)}")
        logger.error(traceback.format_exc())
        return "Error: Failed to fetch quarterly income statement data. Please try again later."

# Define function to fetch quarterly cash flow statement for multiple tickers
def fetch_quarterly_cash_flow_as_text(tickers: List[str]):
    try:
        all_text = ""
        for ticker in tickers:
            ticker = ticker.strip()
            tkr = yf.Ticker(ticker)
            quarterly_cashflow = tkr.quarterly_cashflow
            if quarterly_cashflow is not None and not quarterly_cashflow.empty:
                quarterly_cashflow.reset_index(inplace=True)
                quarterly_cashflow.insert(0, 'Ticker', ticker)
                text = quarterly_cashflow.to_csv(index=False, sep='|')
                all_text += text + "\n"
            else:
                all_text += f"No quarterly cash flow data available for {ticker}\n"
            time.sleep(10)
        return all_text
    except Exception as e:
        logger.error(f"Error fetching quarterly cash flow: {str(e)}")
        logger.error(traceback.format_exc())
        return "Error: Failed to fetch quarterly cash flow data. Please try again later."

# ============================================================
# ROOT ENDPOINT - Minimal response
# ============================================================
@app.get("/")
def root():
    """Root endpoint - minimal response."""
    return {"v": "1.4.0"}


# Define the new endpoint to fetch income statement
@app.get("/get-income-statement/")
@limiter.limit(RATE_LIMIT)
async def get_income_statement_endpoint(request: Request, tickers: str):
    client_ip = get_remote_address(request)
    await check_concurrency(client_ip)
    try:
        tickers_list = tickers.split(",")
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, fetch_income_statement_as_text, tickers_list)
    finally:
        await asyncio.shield(release_concurrency(client_ip))

# Define the new endpoint to fetch cash flow statement
@app.get("/get-cash-flow/")
@limiter.limit(RATE_LIMIT)
async def get_cash_flow_endpoint(request: Request, tickers: str):
    client_ip = get_remote_address(request)
    await check_concurrency(client_ip)
    try:
        tickers_list = tickers.split(",")
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, fetch_cash_flow_as_text, tickers_list)
    finally:
        await asyncio.shield(release_concurrency(client_ip))

# Define the new endpoint to fetch quarterly income statement
@app.get("/get-quarterly-income-statement/")
@limiter.limit(RATE_LIMIT)
async def get_quarterly_income_statement_endpoint(request: Request, tickers: str):
    client_ip = get_remote_address(request)
    await check_concurrency(client_ip)
    try:
        tickers_list = tickers.split(",")
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, fetch_quarterly_income_statement_as_text, tickers_list)
    finally:
        await asyncio.shield(release_concurrency(client_ip))

# Define the new endpoint to fetch quarterly cash flow
@app.get("/get-quarterly-cash-flow/")
@limiter.limit(RATE_LIMIT)
async def get_quarterly_cash_flow_endpoint(request: Request, tickers: str):
    client_ip = get_remote_address(request)
    await check_concurrency(client_ip)
    try:
        tickers_list = tickers.split(",")
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, fetch_quarterly_cash_flow_as_text, tickers_list)
    finally:
        await asyncio.shield(release_concurrency(client_ip))

# Define the existing endpoint to fetch balance sheet
@app.get("/get-balance-sheet/")
@limiter.limit(RATE_LIMIT)
async def get_balance_sheet_endpoint(request: Request, tickers: str):
    client_ip = get_remote_address(request)
    await check_concurrency(client_ip)
    try:
        tickers_list = tickers.split(",")
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, fetch_balance_sheet_as_text, tickers_list)
    finally:
        await asyncio.shield(release_concurrency(client_ip))

# Define the existing endpoint to fetch adjusted closing prices
@app.get("/get-adj-close/")
@limiter.limit(RATE_LIMIT)
async def get_adj_close_endpoint(request: Request, tickers: str, start_date: str, end_date: str):
    client_ip = get_remote_address(request)
    await check_concurrency(client_ip)
    try:
        tickers_list = tickers.split(",")
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, get_adj_close, tickers_list, start_date, end_date)
    finally:
        await asyncio.shield(release_concurrency(client_ip))

# Code to get market capitalization, free float shares, and shares outstanding
@app.get("/get-market-data/", operation_id="get_market_data")
@limiter.limit(RATE_LIMIT)
async def get_market_data(
    request: Request,
    tickers: str = Query(
        description="Comma-separated list of Yahoo Finance ticker symbols (e.g., 'AAPL,MSFT,GOOG')"
    )
):
    """
    Fetches key market data metrics for specified stock tickers from Yahoo Finance.

    This endpoint provides three essential market metrics for each ticker:
    - market_cap: The total market value of all outstanding shares in USD
    - free_float: The number of shares available for public trading (excluding restricted shares)
    - total_shares_outstanding: The total number of shares issued by the company

    These metrics help assess company size, liquidity, and ownership structure.

    Example query: get_market_data?tickers=AAPL,MSFT,GOOG
    """
    client_ip = get_remote_address(request)
    await check_concurrency(client_ip)
    try:
        return await asyncio.get_event_loop().run_in_executor(None, _get_market_data_sync, tickers)
    finally:
        await asyncio.shield(release_concurrency(client_ip))

def _get_market_data_sync(tickers: str):
    tickers_list = tickers.split(",")
    market_data = {}
    try:
        for ticker in tickers_list:
            try:
                logger.info(f"Processing market data for ticker: {ticker}")
                stock = yf.Ticker(ticker)
                info = stock.info
                if not info:
                    logger.warning(f"No info available for ticker: {ticker}")
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
                logger.info(f"Data retrieved for {ticker}: {data}")
                market_data[ticker] = data
                
            except Exception as ticker_error:
                logger.error(f"Error processing ticker {ticker}: {str(ticker_error)}")
                logger.error(traceback.format_exc())
                market_data[ticker] = {
                    "error": "Failed to process ticker",
                    "market_cap": None,
                    "free_float": None,
                    "total_shares_outstanding": None
                }

        return market_data
    except Exception as e:
        logger.error(f"Error in get_market_data: {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": "An error occurred while fetching market data. Please try again later."}

# Define the new function to get all price data
def get_all_prices(tickers: List[str], start_date: str, end_date: str):
    try:
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError as e:
        return {"error": f"Invalid date format: {str(e)}"}

    try:
        logger.info(f"Fetching all price data for tickers: {tickers}")
        logger.info(f"Date range: {start_date} to {end_date}")
        
        # Initialize empty DataFrame to store results
        all_data = pd.DataFrame()
        
        # Fetch data for each ticker
        for ticker_symbol in tickers:
            logger.info(f"Fetching data for {ticker_symbol}")
            ticker = yf.Ticker(ticker_symbol)
            ticker_data = ticker.history(start=start_date, end=end_date)
            
            logger.info(f"Data received for {ticker_symbol}: {ticker_data.shape} rows")
            
            if not ticker_data.empty:
                # Create multi-level column names for each price type
                for column in ticker_data.columns:
                    all_data[f"{ticker_symbol}_{column}"] = ticker_data[column]
        
        if all_data.empty:
            logger.warning("No data available for any ticker")
            return {"error": "No data available for the specified parameters"}
            
        logger.info(f"Final processed data shape: {all_data.shape}")
        
        # Convert to JSON format with date as key and all price types for each ticker
        json_data = {}
        for date_idx in all_data.index:
            # Convert datetime to string in YYYY-MM-DD format
            date_str = date_idx.strftime("%Y-%m-%d")
            json_data[date_str] = {}
            
            for ticker in tickers:
                json_data[date_str][ticker] = {}
                # Add all available price types for this ticker
                for column in ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']:
                    col_name = f"{ticker}_{column}"
                    if col_name in all_data.columns:
                        # Convert numpy types to Python native types
                        value = all_data.at[date_idx, col_name]
                        if pd.isna(value):
                            json_data[date_str][ticker][column] = None
                        else:
                            # Convert numpy types to Python native types
                            if isinstance(value, (np.integer, np.floating)):
                                value = value.item()
                            json_data[date_str][ticker][column] = value
        
        return json_data
        
    except Exception as e:
        logger.error(f"Error in get_all_prices: {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": "An error occurred while fetching price data. Please try again later."}

# TSV format helper for get_all_prices - compact tab-delimited output for LLM/Excel consumption
def get_all_prices_tsv(tickers: List[str], start_date: str, end_date: str):
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError as e:
        return {"error": f"Invalid date format: {str(e)}"}

    try:
        logger.info(f"Fetching all price data (TSV) for tickers: {tickers}")
        all_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
        short_headers = {'Open': 'O', 'High': 'H', 'Low': 'L', 'Close': 'C', 'Volume': 'V', 'Dividends': 'Div', 'Stock Splits': 'Split'}
        multi_ticker = len(tickers) > 1

        # Collect all rows as list of dicts
        rows = []
        for ticker_symbol in tickers:
            ticker_symbol = ticker_symbol.strip()
            logger.info(f"Fetching TSV data for {ticker_symbol}")
            ticker = yf.Ticker(ticker_symbol)
            ticker_data = ticker.history(start=start_dt, end=end_dt)

            if ticker_data.empty:
                continue

            for date_idx in ticker_data.index:
                row = {'Date': date_idx.strftime("%Y-%m-%d")}
                if multi_ticker:
                    row['Ticker'] = ticker_symbol
                for col in all_columns:
                    if col in ticker_data.columns:
                        val = ticker_data.at[date_idx, col]
                        if pd.isna(val):
                            row[col] = ''
                        elif col == 'Volume':
                            row[col] = int(val)
                        else:
                            row[col] = round(float(val), 2)
                    else:
                        row[col] = ''
                rows.append(row)

        if not rows:
            return {"error": "No data available for the specified parameters"}

        # Determine which columns to skip (all zeros or empty)
        skip_cols = set()
        for col in all_columns:
            vals = [r.get(col, '') for r in rows]
            if all(v == '' or v == 0 or v == 0.0 for v in vals):
                skip_cols.add(col)

        # Build header
        keep_cols = [c for c in all_columns if c not in skip_cols]
        header_parts = ['Date']
        if multi_ticker:
            header_parts.append('Ticker')
        header_parts.extend([short_headers[c] for c in keep_cols])
        header_line = '\t'.join(header_parts)

        # Build data lines
        data_lines = [header_line]
        for row in rows:
            parts = [row['Date']]
            if multi_ticker:
                parts.append(row['Ticker'])
            for col in keep_cols:
                parts.append(str(row[col]))
            data_lines.append('\t'.join(parts))

        tsv_string = '\n'.join(data_lines)

        return {
            "data": tsv_string,
            "rows": len(rows),
            "tickers": ','.join(tickers),
            "period": f"{start_date} to {end_date}",
            "format": "tsv",
            "columns": [short_headers[c] for c in keep_cols],
            "zero_cols_dropped": list(skip_cols) if skip_cols else []
        }

    except Exception as e:
        logger.error(f"Error in get_all_prices_tsv: {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": "An error occurred while fetching price data. Please try again later."}


# Define the new endpoint to fetch all price data
@app.get("/get-all-prices/", operation_id="get_all_prices")
@limiter.limit(RATE_LIMIT)
async def get_all_prices_endpoint(
    request: Request,
    tickers: str = Query(
        description="Comma-separated list of Yahoo Finance ticker symbols (e.g., 'AAPL,MSFT,GOOG')"
    ),
    start_date: str = Query(
        description="Start date in YYYY-MM-DD format (e.g., '2023-01-01')"
    ),
    end_date: str = Query(
        description="End date in YYYY-MM-DD format (e.g., '2023-12-31')"
    ),
    format: str = Query(
        default="json",
        description="Response format: 'json' (default) returns nested JSON, 'tsv' returns compact tab-delimited string optimized for LLM context windows (~70% fewer tokens)"
    )
):
    """
    Fetches comprehensive historical price data for specified Yahoo Finance tickers.

    This endpoint retrieves complete historical price data including:
    - Open prices
    - High prices
    - Low prices
    - Close prices
    - Trading volumes
    - Dividend payments
    - Stock splits

    Data is returned in a JSON format organized by date, with each date containing
    data for all requested tickers. For each ticker, all available price metrics
    are provided. This gives a complete picture of price movement and trading activity
    over the specified time period.

    The data can be used for:
    - Technical analysis and charting
    - Performance tracking
    - Dividend history analysis
    - Trading volume analysis
    - Volatility calculations

    Example query: get-all-prices?tickers=AAPL,MSFT&start_date=2023-01-01&end_date=2023-01-31

    TSV format example: get-all-prices?tickers=AAPL&start_date=2023-01-01&end_date=2023-01-31&format=tsv
    TSV response returns a compact tab-delimited string inside JSON, optimized for LLM/Excel consumption.
    """
    client_ip = get_remote_address(request)
    await check_concurrency(client_ip)
    try:
        tickers_list = tickers.split(",")
        loop = asyncio.get_event_loop()
        if format.lower() == "tsv":
            return await loop.run_in_executor(None, get_all_prices_tsv, tickers_list, start_date, end_date)
        return await loop.run_in_executor(None, get_all_prices, tickers_list, start_date, end_date)
    finally:
        await asyncio.shield(release_concurrency(client_ip))

# Define function to fetch detailed company information
def fetch_detailed_info(tickers: List[str]):
    try:
        result = {}
        for ticker in tickers:
            ticker = ticker.strip()
            tkr = yf.Ticker(ticker)
            info = tkr.info
            
            if info:
                # Extract main company info (excluding longBusinessSummary and companyOfficers)
                main_info = {}
                for key, value in info.items():
                    if key not in ['longBusinessSummary', 'companyOfficers']:
                        # Convert numpy types to native Python types
                        if isinstance(value, (np.integer, np.floating)):
                            value = value.item()
                        main_info[key] = value
                
                # Extract company officers info
                officers_info = info.get('companyOfficers', [])
                
                result[ticker] = {
                    'main_info': main_info,
                    'officers': officers_info
                }
            else:
                result[ticker] = {
                    'error': 'No detailed information available'
                }
            time.sleep(10)
        return result
    except Exception as e:
        logger.error(f"Error fetching detailed info: {str(e)}")
        logger.error(traceback.format_exc())
        return {'error': 'Failed to fetch detailed company information. Please try again later.'}

# Define the new endpoint for detailed company information
@app.get("/get-detailed-info/", operation_id="get_detailed_info")
@limiter.limit(RATE_LIMIT)
async def get_detailed_info_endpoint(
    request: Request,
    tickers: str = Query(
        description="Comma-separated list of Yahoo Finance ticker symbols (e.g., 'AAPL,MSFT,GOOG')"
    )
):
    """
    Retrieves comprehensive company information and data points from Yahoo Finance.
    
    Note: This endpoint may take 30 seconds to 3 minutes to respond due to the extensive data being fetched.
    
    This endpoint returns approximately 200 structured data points organized across 11 key categories:
    
    1. Company Information - Basic details about the company, sector, industry, etc.
    2. Stock Price & Market Activity - Current and historical price information
    3. Valuation & Ratios - P/E ratio, PEG, price-to-book, etc.
    4. Dividend Information - Yield, payment history, ex-dividend dates
    5. Earnings & Financials - EPS, revenue, profit margins, etc.
    6. Share Statistics - Float, outstanding shares, short interest
    7. Analyst Price Targets - Consensus targets, recommendations
    8. Technical Indicators - Moving averages, relative strength
    9. Earnings Call / Fiscal Calendar - Upcoming events and dates
    10. Other - Miscellaneous financial metrics and data
    11. Key Executives - Information about company leadership
    
    The data is structured with 'main_info' containing all general data points and 'officers' 
    containing information about key executives.
    
    Example query: get_detailed_info?tickers=AAPL,MSFT,GOOG
    
    Response time: Expect 30 seconds to 3 minutes depending on the number of tickers and data availability.
    """
    client_ip = get_remote_address(request)
    await check_concurrency(client_ip)
    try:
        tickers_list = tickers.split(",")
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, fetch_detailed_info, tickers_list)
    finally:
        await asyncio.shield(release_concurrency(client_ip))

# Define function to fetch income statement in Excel-friendly format
def fetch_income_statement_excel(tickers: List[str]):
    try:
        result = {}
        for ticker in tickers:
            ticker = ticker.strip()
            tkr = yf.Ticker(ticker)
            income_stmt = tkr.income_stmt
            
            if income_stmt is not None and not income_stmt.empty:
                # Reset index to make the metric names a column
                income_stmt.reset_index(inplace=True)
                
                # Convert the data to a dictionary format
                metrics = income_stmt['index'].tolist()
                # Reverse the metrics list to show high-level items first
                metrics.reverse()
                dates = [d.strftime('%Y-%m-%d') if pd.notnull(d) else None 
                        for d in income_stmt.columns[1:]]  # Skip 'index' column
                
                # Create data rows in reversed order
                data = []
                for metric in metrics:
                    row = {'metric': metric}
                    for i, date in enumerate(dates):
                        # Convert numpy types to native Python types
                        # Need to get the original index since we reversed the metrics
                        orig_idx = income_stmt[income_stmt['index'] == metric].index[0]
                        value = income_stmt.iloc[orig_idx, i + 1]
                        if pd.isna(value):
                            row[date] = None
                        elif isinstance(value, (np.integer, np.floating)):
                            row[date] = value.item()
                        else:
                            row[date] = value
                    data.append(row)
                
                result[ticker] = {
                    'dates': dates,
                    'data': data
                }
            else:
                result[ticker] = {
                    'error': 'No income statement data available'
                }
            time.sleep(10)
        return result
    except Exception as e:
        logger.error(f"Error fetching income statement excel: {str(e)}")
        logger.error(traceback.format_exc())
        return {'error': 'Failed to fetch income statement data. Please try again later.'}

# Define the new Excel-specific endpoint for income statement
@app.get("/excel/get-income-statement/", operation_id="get_income_statement_excel")
@limiter.limit(RATE_LIMIT)
async def get_income_statement_excel_endpoint(
    request: Request,
    tickers: str = Query(
        description="Comma-separated list of Yahoo Finance ticker symbols (e.g., 'AAPL,MSFT,GOOG')"
    )
):
    """
    Retrieves annual income statements for specified Yahoo Finance tickers in a structured JSON format.
    
    This endpoint pulls complete annual income statement data from Yahoo Finance with all available line items, including:
    - Total Revenue
    - Cost of Revenue
    - Gross Profit
    - Operating Expenses
    - Operating Income
    - Net Income
    - EPS metrics
    - And many more financial metrics
    
    Data is returned in a well-structured JSON format optimized for analysis and presentation:
    - 'dates': An array of reporting period dates
    - 'data': An array of objects, each containing a financial metric and its values across reporting periods
    
    The structured format makes it easy to:
    - Compare performance across different years
    - Analyze financial trends
    - Calculate financial ratios
    - Generate reports or visualizations
    
    Example query: excel/get-income-statement?tickers=AAPL,MSFT
    """
    client_ip = get_remote_address(request)
    await check_concurrency(client_ip)
    try:
        tickers_list = tickers.split(",")
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, fetch_income_statement_excel, tickers_list)
    finally:
        await asyncio.shield(release_concurrency(client_ip))

# Define function to fetch balance sheet in Excel-friendly format
def fetch_balance_sheet_excel(tickers: List[str]):
    try:
        result = {}
        for ticker in tickers:
            ticker = ticker.strip()
            tkr = yf.Ticker(ticker)
            balance_sheet = tkr.balance_sheet
            
            if balance_sheet is not None and not balance_sheet.empty:
                # Reset index to make the metric names a column
                balance_sheet.reset_index(inplace=True)
                
                # Convert the data to a dictionary format
                metrics = balance_sheet['index'].tolist()
                # Reverse the metrics list to show high-level items first
                metrics.reverse()
                dates = [d.strftime('%Y-%m-%d') if pd.notnull(d) else None 
                        for d in balance_sheet.columns[1:]]  # Skip 'index' column
                
                # Create data rows in reversed order
                data = []
                for metric in metrics:
                    row = {'metric': metric}
                    for i, date in enumerate(dates):
                        # Convert numpy types to native Python types
                        # Need to get the original index since we reversed the metrics
                        orig_idx = balance_sheet[balance_sheet['index'] == metric].index[0]
                        value = balance_sheet.iloc[orig_idx, i + 1]
                        if pd.isna(value):
                            row[date] = None
                        elif isinstance(value, (np.integer, np.floating)):
                            row[date] = value.item()
                        else:
                            row[date] = value
                    data.append(row)
                
                result[ticker] = {
                    'dates': dates,
                    'data': data
                }
            else:
                result[ticker] = {
                    'error': 'No balance sheet data available'
                }
            time.sleep(10)
        return result
    except Exception as e:
        logger.error(f"Error fetching balance sheet excel: {str(e)}")
        logger.error(traceback.format_exc())
        return {'error': 'Failed to fetch balance sheet data. Please try again later.'}

# Define the new Excel-specific endpoint for balance sheet
@app.get("/excel/get-balance-sheet/", operation_id="get_balance_sheet_excel")
@limiter.limit(RATE_LIMIT)
async def get_balance_sheet_excel_endpoint(
    request: Request,
    tickers: str = Query(
        description="Comma-separated list of Yahoo Finance ticker symbols (e.g., 'AAPL,MSFT,GOOG')"
    )
):
    """
    Retrieves annual balance sheets for specified Yahoo Finance tickers in a structured JSON format.
    
    This endpoint pulls complete annual balance sheet data from Yahoo Finance with all available line items, including:
    - Assets (Current Assets, Long-term Assets)
    - Liabilities (Current Liabilities, Long-term Debt)
    - Shareholders' Equity
    - Cash and Equivalents
    - Inventory
    - Accounts Receivable/Payable
    - And many more balance sheet items
    
    Data is returned in a well-structured JSON format optimized for analysis and presentation:
    - 'dates': An array of reporting period dates
    - 'data': An array of objects, each containing a financial metric and its values across reporting periods
    
    The structured format makes it easy to:
    - Assess financial position at different points in time
    - Calculate financial ratios (quick ratio, debt-to-equity, etc.)
    - Evaluate liquidity and solvency
    - Track changes in company financial structure
    
    Example query: excel/get-balance-sheet?tickers=AAPL,MSFT
    """
    client_ip = get_remote_address(request)
    await check_concurrency(client_ip)
    try:
        tickers_list = tickers.split(",")
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, fetch_balance_sheet_excel, tickers_list)
    finally:
        await asyncio.shield(release_concurrency(client_ip))

# Define function to fetch cash flow in Excel-friendly format
def fetch_cash_flow_excel(tickers: List[str]):
    try:
        result = {}
        for ticker in tickers:
            ticker = ticker.strip()
            tkr = yf.Ticker(ticker)
            cash_flow = tkr.cashflow
            
            if cash_flow is not None and not cash_flow.empty:
                # Reset index to make the metric names a column
                cash_flow.reset_index(inplace=True)
                
                # Convert the data to a dictionary format
                metrics = cash_flow['index'].tolist()
                # Reverse the metrics list to show high-level items first
                metrics.reverse()
                dates = [d.strftime('%Y-%m-%d') if pd.notnull(d) else None 
                        for d in cash_flow.columns[1:]]  # Skip 'index' column
                
                # Create data rows in reversed order
                data = []
                for metric in metrics:
                    row = {'metric': metric}
                    for i, date in enumerate(dates):
                        # Convert numpy types to native Python types
                        # Need to get the original index since we reversed the metrics
                        orig_idx = cash_flow[cash_flow['index'] == metric].index[0]
                        value = cash_flow.iloc[orig_idx, i + 1]
                        if pd.isna(value):
                            row[date] = None
                        elif isinstance(value, (np.integer, np.floating)):
                            row[date] = value.item()
                        else:
                            row[date] = value
                    data.append(row)
                
                result[ticker] = {
                    'dates': dates,
                    'data': data
                }
            else:
                result[ticker] = {
                    'error': 'No cash flow data available'
                }
            time.sleep(10)
        return result
    except Exception as e:
        logger.error(f"Error fetching cash flow excel: {str(e)}")
        logger.error(traceback.format_exc())
        return {'error': 'Failed to fetch cash flow data. Please try again later.'}

# Define the new Excel-specific endpoint for cash flow
@app.get("/excel/get-cash-flow/", operation_id="get_cash_flow_excel")
@limiter.limit(RATE_LIMIT)
async def get_cash_flow_excel_endpoint(
    request: Request,
    tickers: str = Query(
        description="Comma-separated list of Yahoo Finance ticker symbols (e.g., 'AAPL,MSFT,GOOG')"
    )
):
    """
    Retrieves annual cash flow statements for specified Yahoo Finance tickers in a structured JSON format.
    
    This endpoint pulls complete annual cash flow data from Yahoo Finance with all available line items, including:
    - Operating Cash Flow
    - Investing Cash Flow
    - Financing Cash Flow
    - Capital Expenditures
    - Dividends Paid
    - Stock Repurchases
    - Free Cash Flow
    - And many more cash flow metrics
    
    Data is returned in a well-structured JSON format optimized for analysis and presentation:
    - 'dates': An array of reporting period dates
    - 'data': An array of objects, each containing a financial metric and its values across reporting periods
    
    The structured format makes it easy to:
    - Analyze cash generation and usage over time
    - Assess operational efficiency
    - Evaluate investment and financing activities
    - Determine sustainability of dividends and share repurchases
    
    Example query: excel/get-cash-flow?tickers=AAPL,MSFT
    """
    client_ip = get_remote_address(request)
    await check_concurrency(client_ip)
    try:
        tickers_list = tickers.split(",")
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, fetch_cash_flow_excel, tickers_list)
    finally:
        await asyncio.shield(release_concurrency(client_ip))

# Define function to fetch quarterly income statement in Excel-friendly format
def fetch_quarterly_income_statement_excel(tickers: List[str]):
    try:
        result = {}
        for ticker in tickers:
            ticker = ticker.strip()
            tkr = yf.Ticker(ticker)
            quarterly_stmt = tkr.quarterly_income_stmt

            if quarterly_stmt is not None and not quarterly_stmt.empty:
                # Reset index to make the metric names a column
                quarterly_stmt.reset_index(inplace=True)

                # Convert the data to a dictionary format
                metrics = quarterly_stmt['index'].tolist()
                # Reverse the metrics list to show high-level items first
                metrics.reverse()
                dates = [d.strftime('%Y-%m-%d') if pd.notnull(d) else None
                        for d in quarterly_stmt.columns[1:]]  # Skip 'index' column

                # Create data rows in reversed order
                data = []
                for metric in metrics:
                    row = {'metric': metric}
                    for i, date in enumerate(dates):
                        # Convert numpy types to native Python types
                        # Need to get the original index since we reversed the metrics
                        orig_idx = quarterly_stmt[quarterly_stmt['index'] == metric].index[0]
                        value = quarterly_stmt.iloc[orig_idx, i + 1]
                        if pd.isna(value):
                            row[date] = None
                        elif isinstance(value, (np.integer, np.floating)):
                            row[date] = value.item()
                        else:
                            row[date] = value
                    data.append(row)

                result[ticker] = {
                    'dates': dates,
                    'data': data
                }
            else:
                result[ticker] = {
                    'error': 'No quarterly income statement data available'
                }
            time.sleep(10)
        return result
    except Exception as e:
        logger.error(f"Error fetching quarterly income statement excel: {str(e)}")
        logger.error(traceback.format_exc())
        return {'error': 'Failed to fetch quarterly income statement data. Please try again later.'}

# Define function to fetch quarterly cash flow in Excel-friendly format
def fetch_quarterly_cash_flow_excel(tickers: List[str]):
    try:
        result = {}
        for ticker in tickers:
            ticker = ticker.strip()
            tkr = yf.Ticker(ticker)
            quarterly_cf = tkr.quarterly_cashflow

            if quarterly_cf is not None and not quarterly_cf.empty:
                # Reset index to make the metric names a column
                quarterly_cf.reset_index(inplace=True)

                # Convert the data to a dictionary format
                metrics = quarterly_cf['index'].tolist()
                # Reverse the metrics list to show high-level items first
                metrics.reverse()
                dates = [d.strftime('%Y-%m-%d') if pd.notnull(d) else None
                        for d in quarterly_cf.columns[1:]]  # Skip 'index' column

                # Create data rows in reversed order
                data = []
                for metric in metrics:
                    row = {'metric': metric}
                    for i, date in enumerate(dates):
                        # Convert numpy types to native Python types
                        # Need to get the original index since we reversed the metrics
                        orig_idx = quarterly_cf[quarterly_cf['index'] == metric].index[0]
                        value = quarterly_cf.iloc[orig_idx, i + 1]
                        if pd.isna(value):
                            row[date] = None
                        elif isinstance(value, (np.integer, np.floating)):
                            row[date] = value.item()
                        else:
                            row[date] = value
                    data.append(row)

                result[ticker] = {
                    'dates': dates,
                    'data': data
                }
            else:
                result[ticker] = {
                    'error': 'No quarterly cash flow data available'
                }
            time.sleep(10)
        return result
    except Exception as e:
        logger.error(f"Error fetching quarterly cash flow excel: {str(e)}")
        logger.error(traceback.format_exc())
        return {'error': 'Failed to fetch quarterly cash flow data. Please try again later.'}

# Define the new Excel-specific endpoint for quarterly income statement
@app.get("/excel/get-quarterly-income-statement/", operation_id="get_quarterly_income_statement_excel")
@limiter.limit(RATE_LIMIT)
async def get_quarterly_income_statement_excel_endpoint(
    request: Request,
    tickers: str = Query(
        description="Comma-separated list of Yahoo Finance ticker symbols (e.g., 'AAPL,MSFT,GOOG')"
    )
):
    """
    Retrieves quarterly income statements for specified Yahoo Finance tickers in a structured JSON format.

    This endpoint pulls complete quarterly income statement data from Yahoo Finance with all available line items, including:
    - Total Revenue
    - Cost of Revenue
    - Gross Profit
    - Operating Expenses
    - Operating Income
    - Net Income
    - EPS metrics
    - And many more financial metrics

    Data is returned in a well-structured JSON format optimized for analysis and presentation:
    - 'dates': An array of quarterly reporting period dates
    - 'data': An array of objects, each containing a financial metric and its values across quarters

    The structured format makes it easy to:
    - Identify seasonal patterns in business performance
    - Track quarter-over-quarter growth rates
    - Analyze short-term business trends
    - Compare quarterly results to analyst expectations
    - Detect turning points in business performance more quickly than annual data

    Example query: excel/get-quarterly-income-statement?tickers=AAPL,MSFT
    """
    client_ip = get_remote_address(request)
    await check_concurrency(client_ip)
    try:
        tickers_list = tickers.split(",")
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, fetch_quarterly_income_statement_excel, tickers_list)
    finally:
        await asyncio.shield(release_concurrency(client_ip))

# Define the new Excel-specific endpoint for quarterly cash flow
@app.get("/excel/get-quarterly-cash-flow/", operation_id="get_quarterly_cash_flow_excel")
@limiter.limit(RATE_LIMIT)
async def get_quarterly_cash_flow_excel_endpoint(
    request: Request,
    tickers: str = Query(
        description="Comma-separated list of Yahoo Finance ticker symbols (e.g., 'AAPL,MSFT,GOOG')"
    )
):
    """
    Retrieves quarterly cash flow statements for specified Yahoo Finance tickers in a structured JSON format.

    This endpoint pulls complete quarterly cash flow data from Yahoo Finance with all available line items, including:
    - Operating Cash Flow
    - Investing Cash Flow
    - Financing Cash Flow
    - Capital Expenditures
    - Dividends Paid
    - Stock Repurchases
    - Free Cash Flow
    - And many more cash flow metrics

    Data is returned in a well-structured JSON format optimized for analysis and presentation:
    - 'dates': An array of quarterly reporting period dates
    - 'data': An array of objects, each containing a financial metric and its values across quarters

    The structured format makes it easy to:
    - Identify seasonal patterns in cash generation
    - Track quarter-over-quarter cash flow trends
    - Analyze short-term liquidity changes
    - Compare quarterly cash flow to analyst expectations
    - Detect changes in capital allocation more quickly than annual data

    Example query: excel/get-quarterly-cash-flow?tickers=AAPL,MSFT
    """
    client_ip = get_remote_address(request)
    await check_concurrency(client_ip)
    try:
        tickers_list = tickers.split(",")
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, fetch_quarterly_cash_flow_excel, tickers_list)
    finally:
        await asyncio.shield(release_concurrency(client_ip))

# Create MCP server and include all relevant endpoints
base_url = os.environ.get("RENDER_EXTERNAL_URL", "http://localhost:8000")
# Ensure protocol is included
if not base_url.startswith(("http://", "https://")):
    base_url = f"http://{base_url}"

mcp = FastApiMCP(
    app,
    name="YFinance MCP API",
    description="MCP server for YFinance data endpoints. Note: Some operations may take up to 3 minutes due to data fetching requirements.",
    include_operations=[
        "get_market_data",
        "get_detailed_info",
        "get_all_prices",
        "get_income_statement_excel",
        "get_balance_sheet_excel",
        "get_cash_flow_excel",
        "get_quarterly_income_statement_excel",
        "get_quarterly_cash_flow_excel"
    ],
    describe_all_responses=True,
    describe_full_response_schema=True,
    # Inject custom httpx client with 5-minute (300 seconds) timeout
    http_client=httpx.AsyncClient(
        timeout=300.0,
        limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        base_url=base_url
    )
)

# Mount the MCP server to the FastAPI app
mcp.mount()

# Log available operations when server starts
@app.on_event("startup")
async def startup_event():
    logger.info("=" * 40)
    logger.info("FastAPI server with MCP integration is starting!")
    logger.info(f"MCP endpoint is available at: {base_url}/mcp")
    logger.info("Using custom httpx client with 5-minute (300 second) timeout")
    logger.info(f"Base URL: {base_url}")
    
    # Log all available routes and their operation IDs
    logger.info("Available routes and operation IDs:")
    for route in app.routes:
        if hasattr(route, "operation_id"):
            logger.info(f"Route: {route.path}, Operation ID: {route.operation_id}")
    
    logger.info("MCP server exposing financial data endpoints from Yahoo Finance")
    logger.info("=" * 40)

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("=" * 40)
    logger.info("FastAPI server is shutting down")
    logger.info("=" * 40)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)