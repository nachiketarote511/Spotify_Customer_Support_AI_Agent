-- dbt test: assert conversations have required fields
-- Ensures data contract compliance in the SQL layer

SELECT conversation_id
FROM {{ ref('int_conversations') }}
WHERE customer_message IS NULL
   OR historical_support_response IS NULL
   OR brand IS NULL
