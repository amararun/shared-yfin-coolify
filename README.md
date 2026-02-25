# YFIN - Yahoo Finance Data API with MCP Integration

FastAPI service providing Yahoo Finance data via REST API + Model Context Protocol (MCP) for AI/LLM interactions.

Live endpoint: [yfin-h.tigzig.com](https://yfin-h.tigzig.com)

## Stack

- Python FastAPI + Uvicorn
- yfinance (Yahoo Finance data)
- FastAPI-MCP 0.3.4 (MCP server integration)
- SlowAPI (rate limiting)
- tigzig-api-monitor (centralized logging)

## Endpoints

| Endpoint | Description |
|----------|-------------|
| `/get-income-statement/` | Annual income statements |
| `/get-balance-sheet/` | Annual balance sheets |
| `/get-cash-flow/` | Annual cash flow statements |
| `/get-quarterly-income-statement/` | Quarterly income statements |
| `/get-quarterly-cash-flow/` | Quarterly cash flow |
| `/get-adj-close/` | Historical adjusted close prices |
| `/get-all-prices/` | Full OHLCV price data |
| `/get-market-data/` | Market cap, float, shares outstanding |
| `/get-detailed-info/` | Company info, executives, valuation ratios |
| `/excel/get-*` | Excel-friendly JSON format versions |
| `/mcp` | MCP endpoint for AI/LLM tool access |

All endpoints accept `?tickers=AAPL,MSFT` query parameter. Price endpoints also accept `start_date` and `end_date`.

## MCP Integration

8 endpoints exposed as MCP tools for AI/LLM consumption: get_market_data, get_detailed_info, get_all_prices, get_income_statement_excel, get_balance_sheet_excel, get_cash_flow_excel, get_quarterly_income_statement_excel, get_quarterly_cash_flow_excel.

See [docs/MCP_UPDATE.md](docs/MCP_UPDATE.md) for the FastAPI-MCP 0.3.4 upgrade guide.

## Setup

```bash
cd mcp_tadata
pip install -r requirements.txt
uvicorn main_mcp:app --host 0.0.0.0 --port 8000
```

## Security Hardening

| # | Layer | What It Does |
|---|-------|-------------|
| 1 | Per-IP rate limiting | 30 requests/minute per IP via SlowAPI, configurable via `RATE_LIMIT` env var |
| 2 | Per-IP concurrency cap | Max 3 simultaneous in-flight requests per IP, configurable via `MAX_CONCURRENT_PER_IP` |
| 3 | Global concurrency cap | Max 6 simultaneous requests server-wide, configurable via `MAX_CONCURRENT_GLOBAL` |
| 4 | CORS credentials disabled | `allow_origins=["*"]` with `allow_credentials=False` — safe for public data API with no auth cookies |
| 5 | Generic error messages | Internal errors return sanitized messages — no stack traces, file paths, or library names leaked |
| 6 | Concurrency leak protection | `asyncio.Shield` on counter release prevents permanent lockout from cancelled requests |
| 7 | Centralized API monitoring | All requests logged via `tigzig-api-monitor` middleware with request body and client IP capture |
| 8 | Custom HTTP client | 5-minute timeout, connection limits (5 keepalive, 10 max) for reliable financial data fetches |

This is a public data API — no user auth (login/signup) or API key required since it serves freely available Yahoo Finance data. For private apps requiring authentication, refer to the [Security Checklist for Web Apps](https://www.tigzig.com/security).

## API Monitoring

All requests are logged via [tigzig-api-monitor](https://pypi.org/project/tigzig-api-monitor/), an open-source centralized logging middleware for FastAPI. The middleware captures request metadata including client IP addresses and request bodies for API monitoring and error tracking.

**Data Retention**: The middleware captures data but does not manage its lifecycle. It is the deployer's responsibility to implement appropriate data retention and deletion policies in accordance with their own compliance requirements (GDPR, CCPA, etc.).

**Graceful Degradation**: If the logging service is unavailable, API calls proceed normally — logging fails silently without affecting functionality.

## Deployment

Docker container deployed via Coolify (Nixpacks) on Hetzner.

## Structure

```
mcp_tadata/
  main_mcp.py        Main application (active code)
  requirements.txt   Python dependencies
docs/
  MCP_UPDATE.md      FastAPI-MCP upgrade guide
```

## License

Apache License 2.0. Uses [yfinance](https://github.com/ranaroussi/yfinance) (Apache 2.0) by Ran Aroussi.

## Author

Built by [Amar Harolikar](https://www.linkedin.com/in/amarharolikar/)

Explore 30+ open source AI tools for analytics, databases & automation at [tigzig.com](https://tigzig.com)
