import math
import pandas as pd
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict
from contextlib import asynccontextmanager
from fastapi import FastAPI
from veyra.serving.model_loader import load_champion


model_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the champion once when the API starts
    pipeline, threshold = load_champion()

    model_state["pipeline"] = pipeline
    model_state["threshold"] = threshold

    print("Veyra API ready!")

    yield

    # Clean up when the API stops
    model_state.clear()


app = FastAPI(
    title="Veyra Fraud Detection API",
    description="Real-time bank account application fraud scoring",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": "pipeline" in model_state,
        "model": "veyra-fraud-detector",
        "alias": "champion",
    }

class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    features: dict[str, object]

@app.post("/predict")
def predict(request: PredictionRequest):
    pipeline = model_state.get("pipeline")
    threshold = model_state.get("threshold")

    if pipeline is None or threshold is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded",
        )

    expected_features = list(
        pipeline.feature_names_in_
    )

    received_features = request.features

    missing = sorted(
        set(expected_features) - set(received_features)
    )
    unexpected = sorted(
        set(received_features) - set(expected_features)
    )

    if missing or unexpected:
        raise HTTPException(
            status_code=422,
            detail={
                "missing_features": missing,
                "unexpected_features": unexpected,
            },
        )

    row = pd.DataFrame(
        [received_features],
        columns=expected_features,
    )

    try:
        probability = float(
            pipeline.predict_proba(row)[0, 1]
        )
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid feature values: {exc}",
        ) from exc

    if not math.isfinite(probability):
        raise HTTPException(
            status_code=500,
            detail="Model returned an invalid probability",
        )

    decision = (
        "manual_review"
        if probability >= threshold
        else "not_flagged"
    )

    return {
        "fraud_probability": probability,
        "decision": decision,
        "threshold": threshold,
        "model": "veyra-fraud-detector",
        "alias": "champion",
    }