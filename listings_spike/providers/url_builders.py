from urllib.parse import urlencode, quote_plus

def rightmove_search_url(address: str, postcode: str) -> str:
    # Use Google site search to land on Rightmove results for this address/postcode
    q = f"site:rightmove.co.uk {address} {postcode} for sale"
    return f"https://www.google.com/search?{urlencode({'q': q})}"

def zoopla_search_url(address: str, postcode: str) -> str:
    q = f"site:zoopla.co.uk {address} {postcode} for sale"
    return f"https://www.google.com/search?{urlencode({'q': q})}"

def generic_map_query(address: str, postcode: str) -> str:
    q = quote_plus(f"{address}, {postcode}")
    return f"https://www.google.com/maps/search/?api=1&query={q}"

