"""
Spark Pipeline — Large Scale Processing
Pipeline B: For large datasets.
Uses PySpark (and optionally Delta Lake).
Produces output matching the same common data contract schema as the Pandas pipeline.

NOTE: This pipeline can run locally with PySpark or on a Databricks cluster.
For local execution, PySpark must be installed.
"""

import os
import sys
import yaml
from typing import Optional

# Add backend root to path (for app.* imports)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


class SparkPipeline:
    """Large-scale data processing pipeline using PySpark."""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), '..', '..', '..', 'configs', 'config.yaml'
            )
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.brand = self.config['brand']
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        self.spark = None
    
    def _init_spark(self):
        """Initialize Spark session."""
        try:
            from pyspark.sql import SparkSession
            
            self.spark = (
                SparkSession.builder
                .appName("HiverAISupport")
                .config("spark.driver.memory", "4g")
                .config("spark.sql.shuffle.partitions", "8")
                .getOrCreate()
            )
            print("Spark session initialized successfully.")
            return True
        except ImportError:
            print("WARNING: PySpark not installed.")
            print("Install with: pip install pyspark")
            print("Falling back to architecture documentation only.")
            return False
        except Exception as e:
            print(f"WARNING: Could not initialize Spark: {e}")
            print("Falling back to architecture documentation only.")
            return False
    
    def run(self, dataset_path: str = None, output_path: str = None):
        """
        Run the Spark processing pipeline.
        
        Steps:
        1. Initialize Spark
        2. Load raw dataset into Spark DataFrame
        3. Filter to brand
        4. Reconstruct conversations using Spark SQL
        5. Validate output schema
        6. Save as Parquet/Delta
        
        Returns:
            Spark DataFrame or None if Spark unavailable
        """
        if dataset_path is None:
            dataset_path = os.path.join(self.project_root, self.config['dataset']['raw_path'])
        
        if output_path is None:
            output_path = os.path.join(self.project_root, 'data', 'processed', 'spark_output')
        
        print("=" * 60)
        print("SPARK PIPELINE: Starting")
        print(f"Brand: {self.brand}")
        print(f"Dataset: {dataset_path}")
        print("=" * 60)
        
        if not self._init_spark():
            self._document_architecture()
            return None
        
        from pyspark.sql import functions as F
        from pyspark.sql.types import StringType, BooleanType, IntegerType
        from pyspark.sql.window import Window
        
        # Step 1: Load data
        print("\n[1/5] Loading dataset into Spark...")
        df = self.spark.read.csv(dataset_path, header=True, inferSchema=True)
        total_rows = df.count()
        print(f"  Loaded {total_rows:,} rows")
        
        # Step 2: Extract @mentions and filter to brand
        print("\n[2/5] Filtering to brand...")
        df = df.withColumn(
            'target_mention',
            F.lower(F.regexp_extract(F.col('text'), r'^@(\w+)', 1))
        )
        df = df.withColumn(
            'author_lower',
            F.lower(F.col('author_id'))
        )
        
        brand = self.brand
        brand_outbound = df.filter(
            (F.col('author_lower') == brand) & (F.col('inbound') == False)
        )
        brand_inbound = df.filter(
            (F.col('target_mention') == brand) & (F.col('inbound') == True)
        )
        
        print(f"  Brand outbound: {brand_outbound.count():,}")
        print(f"  Brand inbound: {brand_inbound.count():,}")
        
        # Step 3: Join inbound with responses
        print("\n[3/5] Reconstructing conversations via join...")
        
        # Self-join: inbound tweet's response_tweet_id -> outbound tweet's tweet_id
        # Parse response_tweet_id (take first value if comma-separated)
        brand_inbound = brand_inbound.withColumn(
            'first_response_id',
            F.split(F.col('response_tweet_id'), ',').getItem(0).cast(IntegerType())
        )
        
        # Join inbound with brand outbound on response link
        conversations = brand_inbound.alias('cust').join(
            brand_outbound.alias('supp'),
            F.col('cust.first_response_id') == F.col('supp.tweet_id'),
            'inner'
        ).select(
            F.md5(
                F.concat(
                    F.col('cust.tweet_id').cast(StringType()),
                    F.lit('_'),
                    F.col('supp.tweet_id').cast(StringType())
                )
            ).substr(1, 12).alias('conversation_id'),
            F.lit(brand).alias('brand'),
            F.regexp_replace(
                F.regexp_replace(F.col('cust.text'), r'^@\w+\s*', ''),
                r'https?://\S+', '[URL]'
            ).alias('customer_message'),
            F.regexp_replace(
                F.regexp_replace(F.col('supp.text'), r'^@\w+\s*', ''),
                r'https?://\S+', '[URL]'
            ).alias('historical_support_response'),
            F.col('cust.created_at').alias('timestamp'),
            F.lit('').alias('conversation_context'),
            F.col('cust.author_id').alias('customer_author'),
            F.col('supp.author_id').alias('support_author'),
            F.col('cust.in_response_to_tweet_id').cast(StringType()).alias('parent_tweet_id'),
            F.col('supp.tweet_id').cast(StringType()).alias('response_tweet_id'),
            F.lit(2).alias('conversation_length'),
        )
        
        # Filter
        print("\n[4/5] Filtering conversations...")
        conversations = conversations.filter(
            (F.length(F.trim(F.col('customer_message'))) > 5) &
            (F.length(F.trim(F.col('historical_support_response'))) > 5) &
            (F.size(F.split(F.col('customer_message'), ' ')) >= 3)
        ).dropDuplicates(['customer_message'])
        
        final_count = conversations.count()
        print(f"  Final conversations: {final_count:,}")
        
        # Step 5: Save
        print("\n[5/5] Saving output...")
        os.makedirs(output_path, exist_ok=True)
        
        # Save as Parquet (universal format)
        parquet_path = os.path.join(output_path, 'conversations.parquet')
        conversations.write.mode('overwrite').parquet(parquet_path)
        print(f"  Saved Parquet to: {parquet_path}")
        
        # Also save as CSV for compatibility
        csv_path = os.path.join(output_path, 'conversations_spark.csv')
        conversations.toPandas().to_csv(csv_path, index=False)
        print(f"  Saved CSV to: {csv_path}")
        
        # Try Delta Lake if available
        try:
            delta_path = os.path.join(output_path, 'conversations_delta')
            conversations.write.format("delta").mode("overwrite").save(delta_path)
            print(f"  Saved Delta to: {delta_path}")
        except Exception:
            print("  Delta Lake not available, skipping Delta format.")
        
        print("\n" + "=" * 60)
        print("SPARK PIPELINE: Complete")
        print(f"  Total conversations: {final_count:,}")
        print("=" * 60)
        
        return conversations
    
    def _document_architecture(self):
        """Document the intended Spark architecture when PySpark is not available."""
        print("\n--- SPARK PIPELINE ARCHITECTURE ---")
        print("""
This pipeline is designed to run on PySpark/Databricks for large-scale processing.

Architecture:
1. Data Ingestion: Read CSV from cloud storage (S3/ADLS/GCS) or local
2. Schema Enforcement: Cast columns to proper types
3. Brand Filtering: Filter using Spark SQL predicates (pushed down)
4. Conversation Reconstruction: 
   - Self-join on response_tweet_id -> tweet_id
   - Window functions for conversation context
5. Text Cleaning: UDFs for @mention removal, URL replacement
6. Deduplication: dropDuplicates on customer_message
7. Output: Save as Delta Lake table with Z-ordering on brand

dbt Integration:
- Raw → Staging: Clean and type-cast
- Staging → Intermediate: Brand filtering and joins
- Intermediate → Mart: Final conversation schema

To run:
  pip install pyspark delta-spark
  python src/pipelines/spark_pipeline.py

For Databricks:
  1. Upload dataset to DBFS
  2. Create cluster with Delta Lake support
  3. Run as notebook or job
        """)
    
    def stop(self):
        """Stop Spark session."""
        if self.spark:
            self.spark.stop()


if __name__ == '__main__':
    pipeline = SparkPipeline()
    result = pipeline.run()
    pipeline.stop()
