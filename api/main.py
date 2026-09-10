"""
FastAPI Application
Customer Support AI Agent API
"""

import os
import sys
import json
import yaml
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

# Load .env BEFORE any component imports that read environment variables
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# Add project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# ------- Request/Response Models -------

class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "1.0.0"
    brand: str = ""

class DatasetAnalysisRequest(BaseModel):
    dataset_path: Optional[str] = None

class DatasetAnalysisResponse(BaseModel):
    file_size_mb: float
    estimated_rows: int
    estimated_columns: int
    classification: str
    recommendation: str

class IntentPredictionRequest(BaseModel):
    message: str
    model_name: str = "baseline"

class IntentPredictionResponse(BaseModel):
    intent: str
    confidence: float
    all_probabilities: Optional[Dict[str, float]] = None

class RetrievalRequest(BaseModel):
    message: str
    top_k: int = 5

class RetrievalResponse(BaseModel):
    results: List[Dict[str, Any]]

class GenerateReplyRequest(BaseModel):
    message: str
    conversation_context: str = ""

class GenerateReplyResponse(BaseModel):
    intent: str
    intent_confidence: float
    retrieved_cases: List[Dict[str, Any]]
    reply: str
    confidence: float
    grounding_summary: str = ""
    should_escalate: bool
    escalation_reason: Optional[str] = None

class EscalationRequest(BaseModel):
    message: str
    intent_confidence: float
    intent_probabilities: Dict[str, float] = {}
    retrieval_results: List[Dict[str, Any]] = []

class EscalationResponse(BaseModel):
    should_escalate: bool
    decision: str
    escalation_score: float
    escalation_reason: Optional[str] = None
    signals: Dict[str, bool] = {}

class SupportAgentRequest(BaseModel):
    message: str
    conversation_context: str = ""

class SupportAgentResponse(BaseModel):
    intent: str
    intent_confidence: float
    retrieved_cases: List[Dict[str, Any]]
    reply: str
    should_escalate: bool
    escalation_reason: Optional[str] = None

# ------- App State -------

class AppState:
    """Holds loaded models and components."""
    def __init__(self):
        self.config = None
        self.intent_classifier = None
        self.retriever = None
        self.rag_generator = None
        self.escalation_engine = None
        self.trivial_baseline = None
        self.simple_baseline = None
        self.initialized = False

state = AppState()

# ------- Lifespan -------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    print("Loading models...")
    
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.yaml')
    with open(config_path) as f:
        state.config = yaml.safe_load(f)
    
    # Load intent classifier
    try:
        from src.intent.train import IntentClassifier
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'models', 'intent_classifier')
        state.intent_classifier = IntentClassifier(model_dir)
        if os.path.exists(os.path.join(model_dir, 'baseline')):
            state.intent_classifier.load_model('baseline')
            print("  Intent classifier loaded (baseline)")
        else:
            print("  Intent classifier not trained yet")
    except Exception as e:
        print(f"  Intent classifier error: {e}")
    
    # Load retriever
    try:
        from src.retrieval.retrieve import HistoricalRetriever
        index_dir = os.path.join(os.path.dirname(__file__), '..', 'models', 'embedding_index')
        state.retriever = HistoricalRetriever(index_dir=index_dir)
        if os.path.exists(os.path.join(index_dir, 'faiss_index.bin')):
            state.retriever.index.load(index_dir)
            state.retriever.embedding_generator.load_model()
            print("  Retriever loaded")
        else:
            print("  FAISS index not built yet")
    except Exception as e:
        print(f"  Retriever error: {e}")
    
    # Load RAG generator
    try:
        from src.rag.generator import RAGGenerator
        state.rag_generator = RAGGenerator(state.config)
        print("  RAG generator ready")
    except Exception as e:
        print(f"  RAG generator error: {e}")
    
    # Load escalation engine
    try:
        from src.escalation.decision import EscalationEngine
        state.escalation_engine = EscalationEngine()
        print("  Escalation engine loaded")
    except Exception as e:
        print(f"  Escalation engine error: {e}")
    
    state.initialized = True
    print("Models loaded. API ready.")
    
    yield
    
    print("Shutting down...")


# ------- App -------

app = FastAPI(
    title="Hiver AI Customer Support Agent",
    description="AI-powered customer support with intent classification, "
                "historical retrieval, RAG generation, and escalation decisions.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static UI files
ui_dir = os.path.join(os.path.dirname(__file__), '..', 'ui')
if os.path.exists(ui_dir):
    app.mount("/ui", StaticFiles(directory=ui_dir, html=True), name="ui")


# ------- Endpoints -------

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        brand=state.config.get('brand', 'unknown') if state.config else 'not loaded'
    )


@app.post("/analyze-dataset", response_model=DatasetAnalysisResponse)
async def analyze_dataset(request: DatasetAnalysisRequest):
    from src.pipelines.pipeline_selector import DatasetAnalyzer
    
    dataset_path = request.dataset_path
    if not dataset_path:
        dataset_path = os.path.join(
            os.path.dirname(__file__), '..', 
            state.config['dataset']['raw_path']
        )
    
    analyzer = DatasetAnalyzer()
    try:
        analysis = analyzer.analyze(dataset_path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    return DatasetAnalysisResponse(**analysis)


@app.post("/predict-intent", response_model=IntentPredictionResponse)
async def predict_intent(request: IntentPredictionRequest):
    if state.intent_classifier is None:
        raise HTTPException(status_code=503, detail="Intent classifier not loaded")
    
    results = state.intent_classifier.predict([request.message], return_proba=True)
    result = results[0]
    return IntentPredictionResponse(**result)


@app.post("/retrieve", response_model=RetrievalResponse)
async def retrieve(request: RetrievalRequest):
    if state.retriever is None or state.retriever.index.index is None:
        raise HTTPException(status_code=503, detail="Retriever not loaded")
    
    results = state.retriever.retrieve(request.message, top_k=request.top_k)
    return RetrievalResponse(results=results)


@app.post("/generate-reply", response_model=GenerateReplyResponse)
async def generate_reply(request: GenerateReplyRequest):
    """Full pipeline: intent → retrieval → RAG → response."""
    errors = []
    
    # Intent prediction with context
    intent = "unknown"
    intent_confidence = 0.0
    intent_probs = {}
    if state.intent_classifier:
        try:
            pred = state.intent_classifier.predict(
                [request.message],
                return_proba=True,
                conversation_context=request.conversation_context
            )[0]
            intent = pred['intent']
            intent_confidence = pred['confidence']
            intent_probs = pred.get('all_probabilities', {})
        except Exception as e:
            errors.append(f"Intent error: {e}")
    
    # Retrieval with context and intent reranking
    retrieved = []
    if state.retriever and state.retriever.index.index is not None:
        try:
            retrieved = state.retriever.retrieve(
                request.message,
                top_k=5,
                conversation_context=request.conversation_context,
                predicted_intent=intent
            )
        except Exception as e:
            errors.append(f"Retrieval error: {e}")
    
    # RAG generation
    reply = ""
    confidence = 0.0
    grounding = ""
    llm_escalate = False
    llm_escalate_reason = None
    
    if state.rag_generator:
        try:
            import asyncio
            result = await asyncio.to_thread(
                state.rag_generator.generate,
                customer_message=request.message,
                predicted_intent=intent,
                intent_confidence=intent_confidence,
                retrieved_cases=retrieved,
                conversation_context=request.conversation_context,
                brand=state.config.get('brand_display_name', 'SpotifyCares'),
            )
            reply = result.get('reply', '')
            confidence = result.get('confidence', 0)
            grounding = result.get('grounding_summary', '')
            llm_escalate = result.get('should_escalate', False)
            llm_escalate_reason = result.get('escalation_reason')
        except Exception as e:
            errors.append(f"RAG error: {e}")
            llm_escalate = True
            llm_escalate_reason = f"Generation error: {str(e)}"
    
    # Authoritative Escalation decision (strict priority hierarchy)
    should_escalate = llm_escalate
    decision = "ESCALATE_TO_HUMAN" if llm_escalate else "AUTO_HANDLE"
    decision_source = "llm_recommendation" if llm_escalate else "policy_and_confidence_gate"
    escalation_reason = llm_escalate_reason
    
    if state.escalation_engine:
        try:
            esc = state.escalation_engine.decide(
                intent_confidence=intent_confidence,
                intent_probabilities=intent_probs,
                retrieval_results=retrieved,
                customer_message=request.message,
                llm_escalation=llm_escalate,
                llm_confidence=confidence,
                llm_escalation_reason=llm_escalate_reason,
            )
            should_escalate = esc['should_escalate']
            decision = esc['decision']
            decision_source = esc['decision_source']
            escalation_reason = esc.get('escalation_reason')
        except Exception as e:
            errors.append(f"Escalation error: {e}")
    
    return GenerateReplyResponse(
        intent=intent,
        intent_confidence=intent_confidence,
        retrieved_cases=retrieved,
        reply=reply,
        confidence=confidence,
        grounding_summary=grounding,
        should_escalate=should_escalate,
        decision=decision,
        decision_source=decision_source,
        escalation_reason=escalation_reason,
    )


@app.post("/decide-escalation", response_model=EscalationResponse)
async def decide_escalation(request: EscalationRequest):
    if state.escalation_engine is None:
        raise HTTPException(status_code=503, detail="Escalation engine not loaded")
    
    result = state.escalation_engine.decide(
        intent_confidence=request.intent_confidence,
        intent_probabilities=request.intent_probabilities,
        retrieval_results=request.retrieval_results,
        customer_message=request.message,
    )
    return EscalationResponse(**result)


@app.post("/support-agent", response_model=SupportAgentResponse)
async def support_agent(request: SupportAgentRequest):
    """Main endpoint - orchestrates the complete support agent pipeline."""
    result = await generate_reply(GenerateReplyRequest(
        message=request.message,
        conversation_context=request.conversation_context,
    ))
    
    return SupportAgentResponse(
        intent=result.intent,
        intent_confidence=result.intent_confidence,
        retrieved_cases=result.retrieved_cases,
        reply=result.reply,
        should_escalate=result.should_escalate,
        decision=result.decision,
        decision_source=result.decision_source,
        escalation_reason=result.escalation_reason,
    )


@app.post("/evaluate")
async def evaluate():
    """Run evaluation on the golden set."""
    golden_path = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'golden', 'golden_eval_set.csv'
    )
    if not os.path.exists(golden_path):
        raise HTTPException(status_code=404, detail="Golden evaluation set not found")
    
    import pandas as pd
    golden_df = pd.read_csv(golden_path)
    
    results = []
    for _, row in golden_df.iterrows():
        msg = row.get('customer_message', '')
        result = await generate_reply(GenerateReplyRequest(message=msg))
        results.append({
            'message': msg,
            'true_intent': row.get('true_intent', ''),
            'predicted_intent': result.intent,
            'intent_confidence': result.intent_confidence,
            'reply': result.reply,
            'should_escalate': result.should_escalate,
        })
    
    return {"results": results, "count": len(results)}


# Serve UI
@app.get("/")
async def root():
    ui_path = os.path.join(os.path.dirname(__file__), '..', 'ui', 'index.html')
    if os.path.exists(ui_path):
        return FileResponse(ui_path)
    return {"message": "Hiver AI Customer Support Agent API", "docs": "/docs"}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
