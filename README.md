# Financial Data API with MCP Integration 📊

> **IMPORTANT NOTICE:** The most up-to-date code is located in the `mcp_tadata/main_mcp.py` file. Please use this file for development and deployment instead of the `main.py` in the root directory.

A FastAPI-based service that fetches financial data from Yahoo Finance, including stock prices, financial statements, and market data. The service includes Model Context Protocol (MCP) integration for AI/LLM interactions.

## AI Co-Analyst Platform

For additional data analysis capabilities, visit our AI Co-Analyst Platform at [rex.tigzig.com](https://rex.tigzig.com). For any questions, reach out to me at amar@harolikar.com

## Server URLs
- **API Server**: [https://yfin.hosting.tigzig.com](https://yfin.hosting.tigzig.com)
- **MCP Endpoint**: [https://yfin.hosting.tigzig.com/mcp](https://yfin.hosting.tigzig.com/mcp)
- **API Documentation**: [https://yfin.hosting.tigzig.com/docs](https://yfin.hosting.tigzig.com/docs)

## MCP Integration
The service integrates the Model Context Protocol (MCP) for seamless AI/LLM interactions:
- Direct access to financial data through natural language queries
- Structured API responses for AI consumption
- Enhanced documentation with operation IDs for tool identification
- Automatic request/response validation and formatting

## Setup & Running Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/amararun/shared-yfin-coolify.git
```
The repository will be cloned into a directory named `shared-yfin-coolify`

### 2. Running Locally
```bash
# Install required packages
pip install -r requirements.txt

# Start the server
uvicorn main:app --reload
```
The API will be available at `http://localhost:8000`
.
### 3. Deployment on Render
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`

## 📖 API Documentation

Interactive API documentation is available at:
- Swagger UI: `/docs`
- ReDoc: `/redoc`

## 🔍 API Endpoints Overview

### Financial Statements

1. **Get Balance Sheet** 
   - Endpoint: `/yfin/get-balance-sheet/`
   - Description: Fetches annual balance sheet data for given companies
   - Example Request:
   ```bash
   curl "http://your-api/yfin/get-balance-sheet/?tickers=AAPL,MSFT"
   ```
   - Returns: Pipe-delimited text format with balance sheet line items

2. **Get Income Statement**
   - Endpoint: `/yfin/get-income-statement/`
   - Description: Retrieves annual income statements
   - Example Request:
   ```bash
   curl "http://your-api/yfin/get-income-statement/?tickers=AAPL,MSFT"
   ```
   - Returns: Pipe-delimited text format with income statement metrics

3. **Get Cash Flow Statement**
   - Endpoint: `/yfin/get-cash-flow/`
   - Description: Provides annual cash flow statements
   - Example Request:
   ```bash
   curl "http://your-api/yfin/get-cash-flow/?tickers=AAPL,MSFT"
   ```
   - Returns: Pipe-delimited text format with cash flow data

### Market Data

4. **Get Adjusted Close Prices** 📈
   - Endpoint: `/yfin/get-adj-close/`
   - Description: Fetches historical adjusted closing prices
   - Example Request:
   ```bash
   curl "http://your-api/yfin/get-adj-close/?tickers=AAPL,MSFT&start_date=2023-01-01&end_date=2023-12-31"
   ```
   - Returns: JSON with daily adjusted closing prices

5. **Get Market Data** 📊
   - Endpoint: `/yfin/get-market-data/`
   - Description: Retrieves market cap, float shares, and shares outstanding
   - Example Request:
   ```bash
   curl "http://your-api/yfin/get-market-data/?tickers=AAPL,MSFT"
   ```
   - Returns: JSON with current market metrics

### Excel-Friendly Endpoints

6. **Excel-Format Financial Statements** 📑
   - Endpoints:
     - `/yfin/excel/get-balance-sheet/`
     - `/yfin/excel/get-income-statement/`
     - `/yfin/excel/get-cash-flow/`
     - `/yfin/excel/get-quarterly-income-statement/`
   - Description: Provides financial statements in Excel-friendly JSON format
   - Example Request:
   ```bash
   curl "http://your-api/yfin/excel/get-balance-sheet/?tickers=AAPL"
   ```
   - Returns: JSON structured for easy Excel integration

### Comprehensive Price Data

7. **Get All Price Data** 📊
   - Endpoint: `/yfin/get-all-prices/`
   - Description: Fetches comprehensive price data including open, high, low, close, volume
   - Example Request:
   ```bash
   curl "http://your-api/yfin/get-all-prices/?tickers=AAPL&start_date=2023-01-01&end_date=2023-12-31"
   ```
   - Returns: JSON with complete daily price information

## 📋 OpenAPI Schema

The API is documented using OpenAPI 3.1.0 specification, which is available in the root directory:

- **File**: `OPENAI_JSON_YFIN.TXT`
- **Format**: OpenAPI 3.1.0 JSON/YAML
- **Purpose**: This schema is specifically designed for integration with ChatGPT and other language models
- **Features**:
  - Complete endpoint documentation
  - Parameter specifications
  - Response schemas
  - Operation IDs for each endpoint
  - Detailed descriptions for better AI understanding

The schema includes all non-Excel endpoints and is optimized for AI integration. You can use this schema to:
- Generate API clients
- Create documentation
- Integrate with ChatGPT and other language models
- Understand the API structure programmatically

## ⚠️ Rate Limiting

The API includes built-in delays between requests to comply with Yahoo Finance's rate limits. Please allow sufficient time between requests.

## 📝 Notes

- All financial statement endpoints return data in either pipe-delimited format or JSON
- Date formats should be in YYYY-MM-DD
- Multiple tickers should be comma-separated
- The API automatically handles CORS for cross-origin requests

## 🔒 Security

The API includes CORS middleware configuration for secure cross-origin requests.

## 📫 Support

For issues and feature requests, please open an issue in the repository.