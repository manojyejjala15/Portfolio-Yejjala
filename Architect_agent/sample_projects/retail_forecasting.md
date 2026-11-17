# Retail Demand Forecasting Platform

A regional retailer with about 200 stores and 20,000 SKUs wants to reduce stockouts and overstock.

Needs:
- Ingest daily sales from each store's POS system (daily file export or REST API).
- Forecast weekly demand per store and SKU for the next 4 weeks.
- Give store managers a simple web dashboard with forecasts and suggested reorder quantities.
- The dashboard should respond in under 2 seconds; the nightly data pipeline should finish within 2 hours.
- Role-based access: store managers see their store, regional managers see their region.

Constraints: Python backend, Streamlit frontend, small team (3 engineers), AWS preferred, budget conscious.
