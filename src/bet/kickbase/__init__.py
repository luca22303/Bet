"""Kickbase: a read-only personal squad/market advisor.

Separate from `bet.ingest` on purpose -- see `client.py`'s module docstring.
Nothing here writes to the point-in-time store; there is no backtesting angle
for your own live fantasy squad, only "what does the market look like right now".
"""
