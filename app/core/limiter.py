from slowapi import Limiter
from slowapi.util import get_remote_address

# Single shared limiter instance — import this wherever you need @limiter.limit()
limiter = Limiter(key_func=get_remote_address)
