import os
from contextlib import asynccontextmanager
from typing import Optional, List
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from src.pipeline.prediction_pipeline import PredictPipeline
from src.logger import logging

# Global prediction pipeline
pipeline: Optional[PredictPipeline] = None


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """Modern FastAPI lifespan context manager for startup and shutdown events."""
    global pipeline
    logging.info("Initializing prediction pipeline on server startup...")
    pipeline = PredictPipeline()
    logging.info("Prediction pipeline loaded successfully.")
    yield


app = FastAPI(
    title="Credit Default Risk Scoring API",
    description="Production-grade ML API for predicting credit default probabilities with SHAP reason codes.",
    version="1.0.0",
    lifespan=lifespan
)

# Initialize templates
templates = Jinja2Templates(directory="templates")


class LoanApplicationRequest(BaseModel):
    AMT_INCOME_TOTAL: float = Field(..., gt=0, description="Annual income of the applicant")
    AMT_CREDIT: float = Field(..., gt=0, description="Requested credit loan amount")
    AMT_ANNUITY: Optional[float] = Field(25000.0, gt=0, description="Loan annuity payment")
    AMT_GOODS_PRICE: Optional[float] = Field(450000.0, gt=0, description="Price of goods/asset")
    DAYS_BIRTH: Optional[int] = Field(-14000, lt=0, description="Applicant age in negative days (-14000 ~ 38 yrs)")
    DAYS_EMPLOYED: Optional[int] = Field(-2000, description="Employment duration in negative days (-2000 ~ 5.5 yrs)")
    NAME_CONTRACT_TYPE: Optional[str] = Field("Cash loans")
    CODE_GENDER: Optional[str] = Field("F")
    FLAG_OWN_CAR: Optional[str] = Field("N")
    FLAG_OWN_REALTY: Optional[str] = Field("Y")
    CNT_CHILDREN: Optional[int] = Field(0, ge=0)
    NAME_INCOME_TYPE: Optional[str] = Field("Working")
    NAME_EDUCATION_TYPE: Optional[str] = Field("Secondary / secondary special")
    NAME_FAMILY_STATUS: Optional[str] = Field("Married")
    NAME_HOUSING_TYPE: Optional[str] = Field("House / apartment")
    EXT_SOURCE_1: Optional[float] = Field(0.5, ge=0.0, le=1.0)
    EXT_SOURCE_2: Optional[float] = Field(0.5, ge=0.0, le=1.0)
    EXT_SOURCE_3: Optional[float] = Field(0.5, ge=0.0, le=1.0)


class RiskReason(BaseModel):
    feature: str
    shap_importance: float
    reason: str


class CreditPredictionResponse(BaseModel):
    probability_of_default: float
    probability_percentage: str
    decision: str
    cutoff_threshold: float
    risk_band: str
    top_risk_reasons: List[RiskReason]


@app.get("/", response_class=HTMLResponse)
def index_page(request: Request):
    """Renders the applicant evaluation form."""
    if os.path.exists("templates/index.html") and os.path.getsize("templates/index.html") > 0:
        return templates.TemplateResponse(request=request, name="index.html")
    return HTMLResponse("<h2>Credit Risk Default Scoring API is Running.</h2><p>Visit <a href='/docs'>/docs</a> for the interactive OpenAPI documentation.</p>")


@app.get("/health")
def health_check():
    """Healthcheck endpoint for container orchestration and uptime monitoring."""
    return {
        "status": "healthy",
        "service": "credit-risk-default-scoring",
        "version": "1.0.0"
    }


@app.post("/predict", response_model=CreditPredictionResponse)
def predict_default(application: LoanApplicationRequest):
    """Calculates applicant default risk, underwriting decision, and regulatory SHAP reason codes."""
    global pipeline
    if pipeline is None:
        pipeline = PredictPipeline()

    payload = application.model_dump()
    result = pipeline.predict(payload)
    return result


@app.post("/predict_ui", response_class=HTMLResponse)
def predict_ui(
    request: Request,
    income: float = Form(...),
    credit: float = Form(...),
    annuity: float = Form(25000.0),
    goods_price: float = Form(450000.0),
    age_years: float = Form(38.0),
    employed_years: float = Form(5.0),
    gender: str = Form("F"),
    contract_type: str = Form("Cash loans"),
    income_type: str = Form("Working"),
    education: str = Form("Secondary / secondary special"),
    ext_source_2: float = Form(0.5),
    ext_source_3: float = Form(0.5)
):
    """Handles HTML form submission and renders results."""
    global pipeline
    if pipeline is None:
        pipeline = PredictPipeline()

    payload = {
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": credit,
        "AMT_ANNUITY": annuity,
        "AMT_GOODS_PRICE": goods_price,
        "DAYS_BIRTH": int(-age_years * 365.25),
        "DAYS_EMPLOYED": int(-employed_years * 365.25),
        "CODE_GENDER": gender,
        "NAME_CONTRACT_TYPE": contract_type,
        "NAME_INCOME_TYPE": income_type,
        "NAME_EDUCATION_TYPE": education,
        "EXT_SOURCE_2": ext_source_2,
        "EXT_SOURCE_3": ext_source_3
    }
    result = pipeline.predict(payload)

    if os.path.exists("templates/result.html") and os.path.getsize("templates/result.html") > 0:
        return templates.TemplateResponse(
            request=request,
            name="result.html",
            context={
                "result": result,
                "inputs": payload
            }
        )
    return JSONResponse(content=result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
