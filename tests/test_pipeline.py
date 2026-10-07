from typing import Generator
import pytest
import pandas as pd
from fastapi.testclient import TestClient

from src.components.data_transformation import add_domain_features
from app import app


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    with TestClient(app) as c:
        yield c


def test_feature_ratios_calculation():
    """Verify that domain credit ratios are accurately computed and guarded against division by zero."""
    sample_df = pd.DataFrame([{
        "AMT_CREDIT": 500000.0,
        "AMT_INCOME_TOTAL": 100000.0,
        "AMT_ANNUITY": 25000.0,
        "AMT_GOODS_PRICE": 450000.0,
        "DAYS_BIRTH": -14610  # exactly 40.0 years
    }])

    enriched = add_domain_features(sample_df)

    assert "CREDIT_INCOME_RATIO" in enriched.columns
    assert "ANNUITY_INCOME_RATIO" in enriched.columns
    assert "ANNUITY_CREDIT_RATIO" in enriched.columns
    assert "GOODS_CREDIT_RATIO" in enriched.columns
    assert "AGE_YEARS" in enriched.columns

    assert round(enriched["CREDIT_INCOME_RATIO"].iloc[0], 2) == 5.0
    assert round(enriched["ANNUITY_INCOME_RATIO"].iloc[0], 2) == 0.25
    assert round(enriched["ANNUITY_CREDIT_RATIO"].iloc[0], 4) == 0.05
    assert round(enriched["GOODS_CREDIT_RATIO"].iloc[0], 2) == 0.90
    assert round(enriched["AGE_YEARS"].iloc[0], 1) == 40.0


def test_api_health(client: TestClient):
    """Verify FastAPI /health endpoint returns HTTP 200 and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "credit-risk-default-scoring"


def test_index_page_loads(client: TestClient):
    """Verify GET / renders the HTML underwriting interface successfully."""
    response = client.get("/")
    assert response.status_code == 200
    assert "AI Credit Underwriting Engine" in response.text


def test_api_predict_flow(client: TestClient):
    """Verify FastAPI /predict endpoint accepts loan payload and produces calibrated PD and SHAP reasons."""
    payload = {
        "AMT_INCOME_TOTAL": 200000.0,
        "AMT_CREDIT": 400000.0,
        "AMT_ANNUITY": 20000.0,
        "AMT_GOODS_PRICE": 350000.0,
        "DAYS_BIRTH": -12000,
        "DAYS_EMPLOYED": -1500,
        "EXT_SOURCE_1": 0.65,
        "EXT_SOURCE_2": 0.60,
        "EXT_SOURCE_3": 0.55,
        "CODE_GENDER": "F",
        "NAME_CONTRACT_TYPE": "Cash loans"
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "probability_of_default" in data
    assert 0.0 <= data["probability_of_default"] <= 1.0
    assert data["decision"] in ["Approved", "Rejected"]
    assert "risk_band" in data
    assert "cutoff_threshold" in data
    assert isinstance(data["top_risk_reasons"], list)
    assert len(data["top_risk_reasons"]) >= 1
