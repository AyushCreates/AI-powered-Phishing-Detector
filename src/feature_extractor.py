import re
import math
from urllib.parse import urlparse
import tldextract

def url_entropy(s: str) -> float:
    """Shannon entropy of a string (for randomness in domain)."""
    from collections import Counter
    cnt = Counter(s)
    probs = [v / len(s) for v in cnt.values()]
    return -sum(p * math.log2(p) for p in probs)

def extract_features(url: str) -> dict:
    """Extract basic lexical features from a URL."""
    p = urlparse(url.strip())
    ext = tldextract.extract(url)
    domain = ext.domain + ('.' + ext.suffix if ext.suffix else '')
    path = p.path or ''

    features = {
        "url_len": len(url),
        "domain_len": len(domain),
        "path_len": len(path),
        "num_digits": sum(c.isdigit() for c in url),
        "num_hyphen": url.count('-'),
        "num_at": url.count('@'),
        "num_query": url.count('?'),
        "has_ip": bool(re.search(r'//\d+\.\d+\.\d+\.\d+', url)),
        "entropy": url_entropy(domain),
        "tokens": len(re.split(r'[\./\-_\?=&]+', url))
    }
    return features
