# Backward compatibility shim: older code imports `from utils import vectorstore` but file is util.py
from util import vectorstore, get_vectorstore

__all__ = ["vectorstore", "get_vectorstore"]
