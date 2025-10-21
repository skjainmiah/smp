# src/holidays.py
"""
Simple trading day checker for India.
Tries python 'holidays' package first. Falls back to simple Mon-Fri check.
"""
from datetime import date
import logging

LOG = logging.getLogger("holidays")
try:
    import holidays as pyholidays
    _INDIA = pyholidays.India()
except Exception:
    _INDIA = None
    LOG.warning("python 'holidays' package not available. Install via: pip install holidays for accurate holiday detection")

def is_trading_day(d):
    """
    d: date or datetime.date
    returns True if trading day (not weekend and not holiday)
    """
    try:
        if not hasattr(d, "weekday"):
            d = d.date()
        wd = d.weekday()
        if wd >= 5:
            return False
        if _INDIA:
            return d not in _INDIA
        # fallback simple Mon-Fri
        return True
    except Exception:
        LOG.exception("is_trading_day failed for %s", d)
        return True
