# Data Pipeline — Hiver AI Customer Support Agent

Contains dbt models and data contract schemas for the data transformation layer.

## dbt Project

The dbt project defines SQL transformations for data warehouse processing:

- **`stg_raw_tweets.sql`** — Staging: clean and type-cast raw tweets
- **`int_conversations.sql`** — Intermediate: brand filtering and conversation joins
- **`assert_contract_compliance.sql`** — Test: validates data contract compliance

## Data Contract

The `schemas/data_contract.py` defines the expected schema for processed data.

## Running dbt (Production)

```bash
cd data-pipeline/dbt
dbt run --profiles-dir .
dbt test
```

> **Note**: This is an architectural placeholder. In production, this would connect to a Databricks/Spark cluster.
