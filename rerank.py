from cohere_api import rerank

def rerank_retrievals(query, retrievals, n):
  """Retrieves the top n reranked out of retrieved documents."""
  
  # Call the rerank function with the provided query, retrievals, and n
  responses = rerank(
    query = query,         # The search query
    retrievals = retrievals, # The list of retrieved documents
    n = n,                 # The number of top documents to retrieve
  ).results
  
  # Extract and return the text of the top n reranked documents
  return [result.document["text"] for result in responses]