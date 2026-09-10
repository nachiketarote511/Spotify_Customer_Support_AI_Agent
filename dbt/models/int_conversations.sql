-- dbt model: int_conversations
-- Reconstructs customer-support conversation chains.
-- Joins inbound tweets with their outbound responses.

{{ config(materialized='table') }}

WITH inbound AS (
    SELECT *
    FROM {{ ref('stg_raw_tweets') }}
    WHERE is_inbound = TRUE
),

outbound AS (
    SELECT *
    FROM {{ ref('stg_raw_tweets') }}
    WHERE is_inbound = FALSE
),

conversations AS (
    SELECT
        i.tweet_id AS conversation_id,
        o.brand_id AS brand,
        i.tweet_text AS customer_message,
        o.tweet_text AS historical_support_response,
        i.created_at AS timestamp,
        i.author_id AS customer_author,
        o.author_id AS support_author,
        i.in_response_to_tweet_id AS parent_tweet_id,
        i.response_tweet_id
    FROM inbound i
    LEFT JOIN outbound o
        ON i.response_tweet_id = o.tweet_id
    WHERE o.tweet_text IS NOT NULL
)

SELECT * FROM conversations
