# scheduler helper: use system cron / systemd / schedule library
import schedule
import time
from .yfinance_fetch import fetch_and_store_symbol

def daily_job(symbols_file='data/meta/symbols.txt'):
    with open(symbols_file) as f:
        syms = [s.strip() for s in f if s.strip()]
    for s in syms:
        try:
            fetch_and_store_symbol(s)
        except Exception as e:
            print("Error fetching", s, e)

if __name__ == '__main__':
    schedule.every().day.at("06:00").do(daily_job)
    while True:
        schedule.run_pending()
        time.sleep(30)
