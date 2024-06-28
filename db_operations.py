# db_operations.py
from sqlalchemy.orm import Session
from models import Stock, SessionLocal

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def add_stock(db: Session, stock_data):
    stock = Stock(
        date=stock_data['date'],
        open=stock_data['open'],
        high=stock_data['high'],
        low=stock_data['low'],
        close=stock_data['close'],
        volume=stock_data['volume'],
        ticker=stock_data['ticker']
    )
    db.add(stock)
    db.commit()
    db.refresh(stock)
    return stock

def get_all_stocks(db: Session):
    return db.query(Stock).all()
