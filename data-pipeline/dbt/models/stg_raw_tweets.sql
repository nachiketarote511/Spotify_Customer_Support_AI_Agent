-- dbt model: stg_raw_tweets
-- Stages raw tweet data from the CSV into normalized columns.
-- In production, this reads from a Delta Lake table.

{{ config(materialized='view') }}

SELECT
    CAST(tweet_id AS VARCHAR) AS tweet_id,
    CAST(author_id AS VARCHAR) AS author_id,
    CAST(inbound AS BOOLEAN) AS is_inbound,
    created_at AS created_at,
    text AS tweet_text,
    response_tweet_id,
    in_response_to_tweet_id,
    -- Extract brand from author_id for outbound tweets
    CASE
        WHEN CAST(inbound AS BOOLEAN) = FALSE THEN CAST(author_id AS VARCHAR)
        ELSE NULL
    END AS brand_id
FROM {{ source('raw', 'twcs') }}
WHERE text IS NOT NULL
