"""The sans-I/O protocol layer: request builders, response readers and multi-step flows.

Nothing here performs I/O. The sync and async clients drive the same code, which is what keeps
them in step.
"""
