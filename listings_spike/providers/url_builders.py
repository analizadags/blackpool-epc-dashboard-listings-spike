from urllib.parse import urlencode, quote_plus

def _google(q: str) -> str:
    return f"https://www.google.com/search?{urlencode({'q': q})}"

def rightmove_search_url(address: str, postcode: str) -> str:
    # Specific query for this address + postcode
    q_exact = f"site:rightmove.co.uk {address} {postcode} for sale"
    return _google(q_exact)

def zoopla_search_url(address: str, postcode: str) -> str:
    q_exact = f"site:zoopla.co.uk {address} {postcode} for sale"
    return _google(q_exact)

def generic_map_query(address: str, postcode: str) -> str:
    q = quote_plus(f"{address}, {postcode}")
    return f"https://www.google.com/maps/search/?api=1&query={q}"

# Broader fallback queries (street-level or postcode only)
def rightmove_broad_url(street_or_block: str, postcode: str) -> str:
    return _google(f"site:rightmove.co.uk {street_or_block} {postcode} for sale")

def zoopla_broad_url(street_or_block: str, postcode: str) -> str:
    return _google(f"site:zoopla.co.uk {street_or_block} {postcode} for sale")
