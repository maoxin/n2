from datetime import timedelta, datetime
from dateutil.parser import parse as parse_date

import pandas as pd


def str2date(date_str):
    return parse_date(str(date_str))


def date2str(date, format="%Y%m%d%H%M%S"):
    return to_date(date).strftime(format)


def to_date(date):
    if not isinstance(date, datetime):
        date = str2date(str(date))
    return date


def to_datestr(date, format="%Y%m%d%H%M%S"):
    return date2str(to_date(date), format)


def get_date_range(start_date, end_date, to_str=False, format="%Y%m%d%H%M%S"):
    start_date = to_date(start_date)
    end_date = to_date(end_date)
    date_range = list(pd.date_range(start_date, end_date))
    if to_str:
        date_range = [to_datestr(date, format) for date in date_range]
    return date_range


def get_date_nday_before(date, days=0, hours=0, minutes=0, seconds=0, to_str=False, format="%Y%m%d%H%M%S"):
    date = to_date(date)
    date_before = date - timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
    if to_str:
        date_before = to_datestr(date_before, format)
    return date_before
