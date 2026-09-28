"""Connectivity-only search diagnostics; prints no credentials."""
import time
import httpx
cases = [
 ('arxiv-simple','https://export.arxiv.org/api/query', {'search_query':'all:multimodal','max_results':3}),
 ('arxiv-html','https://arxiv.org/search/', {'query':'multimodal reasoning','searchtype':'all','abstracts':'show','order':'-announced_date_first','size':50}),
 ('semantic-scholar','https://api.semanticscholar.org/graph/v1/paper/search', {'query':'multimodal reasoning','limit':5,'fields':'title,abstract,externalIds,authors,year'}),
 ('crossref','https://api.crossref.org/works', {'query':'multimodal reasoning','rows':3}),
]
with httpx.Client(timeout=20, follow_redirects=True) as client:
    for name, url, params in cases:
        try:
            r=client.get(url, params=params)
            print(name, r.status_code, r.headers.get('content-type'), len(r.content), r.text[:100].replace('\n',' '), flush=True)
        except httpx.RequestError as exc:
            print(name, type(exc).__name__, flush=True)
        time.sleep(3.1)
