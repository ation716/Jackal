# -*- coding: utf-8 -*-
# This file is kept for backward compatibility only.
# All logic has been moved to limit_tracker.py.
from limit_tracker import LimitTracker, is_main_board, is_non_st, get_trading_days  # noqa: F401

if __name__ == '__main__':
    tracker = LimitTracker(days=15)
    tracker.run(verbose=True)
    tracker.print_summary()
