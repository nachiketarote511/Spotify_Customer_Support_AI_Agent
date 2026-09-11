import os
import pandas as pd
import json

def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    old_golden_path = os.path.join(root_dir, 'data', 'golden', 'golden_eval_set.csv')
    
    new_golden_dir = os.path.join(root_dir, 'backend', 'data', 'golden')
    os.makedirs(new_golden_dir, exist_ok=True)
    
    new_golden_path = os.path.join(new_golden_dir, 'golden_eval_set.csv')
    metadata_path = os.path.join(new_golden_dir, 'golden_eval_set_metadata.json')
    annotations_path = os.path.join(new_golden_dir, 'golden_human_annotations.csv')
    
    # Load old dataset
    df = pd.read_csv(old_golden_path)
    
    # New schema mapping
    # Schema should keep these separate:
    # machine_suggested_intent
    # human_intent
    # machine_suggested_escalation
    # human_escalation
    # machine_reason
    # human_reason
    # human_reviewed
    # reviewer_notes
    
    # Drop old human fields if they exist to start fresh according to the new schema
    columns_to_keep = [
        'conversation_id', 'customer_message', 'true_intent', 'conversation_context', 
        'historical_response', 'difficulty', 'sampling_category', 'machine_suggested_intent'
    ]
    
    # Keep only relevant columns that existed before
    df = df[[col for col in columns_to_keep if col in df.columns]]
    
    # Add new schema columns
    df['machine_suggested_escalation'] = False  # Placeholder, will be updated by ML in a real scenario
    df['machine_reason'] = "AI-assisted draft — human review pending."
    df['human_intent'] = None
    df['human_escalation'] = None
    df['human_reason'] = None
    df['human_reviewed'] = False
    df['reviewer_notes'] = None
    
    # Save the updated golden set
    df.to_csv(new_golden_path, index=False)
    print(f"Created {new_golden_path} with {len(df)} examples.")
    
    # Create empty annotations file with the same schema, if not exists
    if not os.path.exists(annotations_path):
        # Annotations usually track just the completed ones, or we can just use golden_human_annotations.csv
        # as a copy that gets updated. The requirement says:
        # "Save final human annotations to: backend/data/golden/golden_human_annotations.csv"
        # We can just initialize it with the same structure, or empty.
        df.to_csv(annotations_path, index=False)
        print(f"Created {annotations_path}")
    
    # Create metadata
    metadata = {
        "dataset_name": "SpotifyCares Golden Evaluation Set",
        "description": "200 real examples sampled from Twitter dataset for evaluation. AI-assisted draft labels generated.",
        "size": len(df),
        "sampling_strategy": "conversation-level stratified sampling across intents to prevent leakage",
        "status": "200 real examples were sampled and AI-assisted draft labels were generated. Human review is tracked separately.",
        "schema_version": "2.0",
        "fields": {
            "machine_suggested_intent": "AI prediction for intent",
            "human_intent": "Human reviewed ground truth intent",
            "machine_suggested_escalation": "AI prediction for escalation (boolean)",
            "human_escalation": "Human reviewed escalation decision (boolean)",
            "machine_reason": "AI reasoning for its suggestions",
            "human_reason": "Human reasoning for their decision",
            "human_reviewed": "Boolean flag indicating if a human has reviewed this record",
            "reviewer_notes": "Additional context from the human reviewer"
        }
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Created {metadata_path}")

if __name__ == '__main__':
    main()
