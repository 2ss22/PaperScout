"""Test the screenshot's old query through the current recovery path."""
from research_agent.tools import ArxivClient

query = 'all:("multimodal large language model" OR "vision-language model" OR "large vision-language model") AND all:("multi-step reasoning" OR "visual reasoning" OR "chain-of-thought")'
client = ArxivClient()
try:
    papers = client.search(query, 3)
    print('papers:', len(papers), flush=True)
    print('attempts:', client.search_diagnostics, flush=True)
    for paper in papers:
        print(paper.id, paper.title, paper.search_source, flush=True)
finally:
    client.close()
