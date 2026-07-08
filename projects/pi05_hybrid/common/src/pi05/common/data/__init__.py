"""Shared data helpers.

Import heavyweight helpers such as normalization from their concrete modules.
This keeps lightweight config and codec imports usable before torch is loaded.
"""
