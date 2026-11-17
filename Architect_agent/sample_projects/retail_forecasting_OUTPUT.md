# Retail Forecasting: Technical Architecture

**Review status:** Approved (score 8/10 after 2 review round(s))

## 1. Summary
A web platform that forecasts weekly product demand per store and recommends reorder quantities.

## 2. Requirements

### Functional
| ID | Requirement | Priority |
|---|---|---|
| FR-1 | Ingest daily sales from store POS systems | must |
| FR-2 | Generate 4-week demand forecasts per SKU and store | must |
| FR-3 | Show forecasts and reorder suggestions in a dashboard | must |

### Non-functional
| ID | Requirement | Priority |
|---|---|---|
| NFR-1 | Dashboard p95 response under 2 s | must |
| NFR-2 | Nightly pipeline finishes within 2 hours | should |

### Constraints
- Python backend
- Streamlit frontend

### Assumptions
- About 200 stores and 20,000 SKUs
- POS exports are available as daily files or an API

### Open questions
- Is real-time inventory needed, or is daily sufficient?

## 3. Architecture (Modular monolith API plus scheduled batch pipeline)
A FastAPI service serves forecasts to a Streamlit dashboard. A nightly Airflow pipeline loads POS data into PostgreSQL, builds features, trains models and publishes forecasts.

| Component | Layer | Responsibility | Tech | Depends on |
|---|---|---|---|---|
| Dashboard | frontend | Forecast and reorder views | Streamlit | API Service |
| API Service | backend | AuthN/Z, forecast and inventory endpoints | FastAPI | Forecast Store, Forecast Cache |
| Forecast Engine | backend | Trains models and writes forecasts | Python, scikit-learn, XGBoost | Feature Tables, Model Registry, Forecast Store |
| POS Connector | integration | Pulls daily sales from POS exports | Python, boto3 | - |
| Pipeline Scheduler | integration | Orchestrates nightly ingest and training | Airflow | POS Connector, Feature Tables, Forecast Engine |
| Feature Tables | data | Cleaned sales and calendar features | PostgreSQL | - |
| Forecast Store | data | Published forecasts and reorder suggestions | PostgreSQL | - |
| Model Registry | data | Versioned model artifacts and metrics | MLflow | - |
| Monitoring | infra | Metrics, logs and alerts | Prometheus, Grafana | - |
| Forecast Cache | data | Read-through cache for hot forecasts | Redis | - |

## 4. Diagrams

### Component architecture
```mermaid
flowchart LR
  subgraph frontend["Frontend"]
    c_Dashboard["Dashboard<br/>Streamlit"]
  end
  subgraph backend["Backend"]
    c_API_Service["API Service<br/>FastAPI"]
    c_Forecast_Engine["Forecast Engine<br/>Python, scikit-learn, XGBoost"]
  end
  subgraph integration["Integration"]
    c_POS_Connector["POS Connector<br/>Python, boto3"]
    c_Pipeline_Scheduler["Pipeline Scheduler<br/>Airflow"]
  end
  subgraph data["Data"]
    c_Feature_Tables["Feature Tables<br/>PostgreSQL"]
    c_Forecast_Store["Forecast Store<br/>PostgreSQL"]
    c_Model_Registry["Model Registry<br/>MLflow"]
    c_Forecast_Cache["Forecast Cache<br/>Redis"]
  end
  subgraph infra["Infrastructure"]
    c_Monitoring["Monitoring<br/>Prometheus, Grafana"]
  end
  c_Dashboard --> c_API_Service
  c_API_Service --> c_Forecast_Store
  c_API_Service --> c_Forecast_Cache
  c_Forecast_Engine --> c_Feature_Tables
  c_Forecast_Engine --> c_Model_Registry
  c_Forecast_Engine --> c_Forecast_Store
  c_Pipeline_Scheduler --> c_POS_Connector
  c_Pipeline_Scheduler --> c_Feature_Tables
  c_Pipeline_Scheduler --> c_Forecast_Engine
```

### Flow: View forecast
```mermaid
sequenceDiagram
  participant p_User as User
  participant p_Dashboard as Dashboard
  participant p_API_Service as API Service
  participant p_Forecast_Store as Forecast Store
  p_User->>p_Dashboard: Select store and SKU
  p_Dashboard->>p_API_Service: GET /forecasts
  p_API_Service->>p_Forecast_Store: Query latest forecast
  p_API_Service->>p_Dashboard: Forecast JSON
```

### Flow: Nightly pipeline
```mermaid
sequenceDiagram
  participant p_Pipeline_Scheduler as Pipeline Scheduler
  participant p_POS_Connector as POS Connector
  participant p_Feature_Tables as Feature Tables
  participant p_Forecast_Engine as Forecast Engine
  participant p_Forecast_Store as Forecast Store
  p_Pipeline_Scheduler->>p_POS_Connector: Fetch daily sales
  p_POS_Connector->>p_Feature_Tables: Upsert cleaned sales
  p_Pipeline_Scheduler->>p_Forecast_Engine: Train and predict
  p_Forecast_Engine->>p_Forecast_Store: Write forecasts
```

## 5. Deployment
- Docker images per service on a managed container platform
- Managed PostgreSQL with daily backups
- CI pipeline runs tests and builds images

## 6. Decisions (ADRs)

**ADR-1: Batch forecasting instead of streaming**
- Decision: Train and predict nightly
- Rationale: Reorders are planned daily; batch is simpler and cheaper
- Alternatives considered: Streaming with Kafka

**ADR-2: Add Redis read cache**
- Decision: Cache forecast reads for 15 minutes
- Rationale: Meets the 2 s p95 target under dashboard load
- Alternatives considered: Materialized views

## 7. Risks
| Risk | Mitigation |
|---|---|
| POS data arrives late or malformed | Schema validation, data-quality checks and alerts before training |

## 8. Review findings
| Severity | Issue | Suggestion |
|---|---|---|
| low | Define data retention for old forecasts | Archive forecasts older than 13 months |
