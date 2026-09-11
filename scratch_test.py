import sys
sys.path.insert(0, 'k:/Hiver_assignment/Customer_Support_AI_Agent/backend')
try:
    from app.intent.train import IntentClassifier
    ic = IntentClassifier('k:/Hiver_assignment/Customer_Support_AI_Agent/models/intent_classifier')
    ic.load_model('baseline')
    res = ic.predict(['My app is crashing'])
    print('Predict result:', res)
except Exception as e:
    import traceback
    traceback.print_exc()
