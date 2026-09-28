"""Read-only connectivity diagnostics; never reads model keys."""
import time

import httpx

URLS = [
    'https://arxiv.org/api/query?search_query=ti:coconut&max_results=3',
    'https://export.arxiv.org/api/query?id_list=2412.06769',
    'https://export.arxiv.org/api/query?search_query=all:electron&max_results=1',
    'https://arxiv.org/abs/2412.06769',
]
with httpx.Client(timeout=25, follow_redirects=True, headers={'User-Agent': 'PaperScout/0.1'}) as client:
    for url in URLS:
        try:
            response = client.get(url)
            print(response.status_code, str(response.url), response.headers.get('server'), flush=True)
            if response.status_code >= 400:
                print(response.text[:200], flush=True)
        except httpx.RequestError as exc:
            print(type(exc).__name__, flush=True)
        time.sleep(3.1)
