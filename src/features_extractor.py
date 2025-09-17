import numpy as np
from urllib.parse import urlparse
import re

def extract_features(url):
    """
    Extracts 48 numerical features from a URL for phishing detection.
    """

    features = []

    parsed = urlparse(url)
    domain = parsed.netloc
    path = parsed.path
    query = parsed.query

    # 1. URL length
    features.append(len(url))

    # 2. Number of dots
    features.append(url.count('.'))

    # 3. Number of hyphens
    features.append(url.count('-'))

    # 4. Number of digits
    features.append(sum(c.isdigit() for c in url))

    # 5. Presence of '@'
    features.append(int('@' in url))

    # 6. Presence of HTTPS
    features.append(int(url.startswith("https")))

    # 7. Number of query parameters
    features.append(len(query.split('&')) if query else 0)

    # 8. Length of domain
    features.append(len(domain))

    # 9. Number of subdomains
    features.append(len(domain.split('.')) - 1 if domain else 0)

    # 10. IP address in domain
    features.append(int(re.match(r'^\d{1,3}(\.\d{1,3}){3}$', domain) is not None))

    # 11. Length of path
    features.append(len(path))

    # 12. Number of directories in path
    features.append(path.count('/'))

    # 13. Number of suspicious keywords in URL
    suspicious_keywords = ['secure', 'login', 'verify', 'account', 'update', 'free', 'bonus']
    features.append(sum(keyword in url.lower() for keyword in suspicious_keywords))

    # 14. Number of special characters
    special_chars = ['?', '=', '&', '%', '$', '#']
    features.append(sum(url.count(c) for c in special_chars))

    # 15. Ratio of digits to letters
    letters = sum(c.isalpha() for c in url)
    digits = sum(c.isdigit() for c in url)
    features.append(digits / letters if letters > 0 else 0)

    # 16. Presence of double slash in path (after domain)
    features.append(int('//' in path))

    # 17. Presence of suspicious TLDs (example: .xyz, .top)
    suspicious_tlds = ['.xyz', '.top', '.club', '.info', '.review']
    features.append(int(any(domain.endswith(tld) for tld in suspicious_tlds)))

    # 18. Count of '@' symbols
    features.append(url.count('@'))

    # 19. Number of subdirectories in path
    features.append(path.count('/'))

    # 20. Presence of dash in domain
    features.append(int('-' in domain))

    # 21. Number of underscores
    features.append(url.count('_'))

    # 22. Length of TLD
    tld = domain.split('.')[-1] if '.' in domain else ''
    features.append(len(tld))

    # 23. Number of fragments (#)
    features.append(url.count('#'))

    # 24. Number of '=' signs
    features.append(url.count('='))

    # 25. Number of '?' signs
    features.append(url.count('?'))

    # 26. Number of '&' signs
    features.append(url.count('&'))

    # 27. Number of '%' signs
    features.append(url.count('%'))

    # 28. Number of '$' signs
    features.append(url.count('$'))

    # 29. Number of uppercase letters
    features.append(sum(c.isupper() for c in url))

    # 30. Number of lowercase letters
    features.append(sum(c.islower() for c in url))

    # 31. Number of digits in domain
    features.append(sum(c.isdigit() for c in domain))

    # 32. Number of letters in domain
    features.append(sum(c.isalpha() for c in domain))

    # 33. Length of last directory in path
    last_dir = path.split('/')[-1] if path else ''
    features.append(len(last_dir))

    # 34. Presence of IP in path
    features.append(int(re.search(r'\d{1,3}(\.\d{1,3}){3}', path) is not None))

    # 35. Presence of port number in URL
    features.append(int(':' in domain))

    # 36. Number of dots in path
    features.append(path.count('.'))

    # 37. Count of repeated characters (like 'aa', 'bb', etc.)
    repeated_chars = sum(1 for i in range(len(url)-1) if url[i] == url[i+1])
    features.append(repeated_chars)

    # 38. Number of numeric tokens in path
    numeric_tokens = sum(1 for token in path.split('/') if token.isdigit())
    features.append(numeric_tokens)

    # 39. Presence of encoded characters (%)
    features.append(int('%' in url))

    # 40. Number of subdomains that are suspicious keywords
    features.append(sum(keyword in domain.lower() for keyword in suspicious_keywords))

    # 41–48: Placeholder extra features
    for _ in range(48 - len(features)):
        features.append(0)

    return np.array(features).reshape(1, -1)
