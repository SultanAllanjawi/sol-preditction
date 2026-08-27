from fastapi import APIRouter

from data_manager import get_combined_news, get_forexfactory_calendar
from tickers import normalize

router = APIRouter(prefix="/api", tags=["news"])


@router.get("/news/{ticker}")
def news(ticker: str):
    return get_combined_news(normalize(ticker))


@router.get("/calendar")
def calendar():
    return {"events": get_forexfactory_calendar()}
