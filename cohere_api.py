import cohere
from dotenv import load_dotenv
from os import getenv

load_dotenv()
API_KEY = getenv("COHERE_API_KEY")
co = cohere.Client(API_KEY)

def embed(docs, input_type):
    return co.embed(
      texts=docs,
      model='embed-english-v3.0',
      input_type=input_type
    )

def tokenize(text):
    return co.tokenize(
      text=text,
      model='embed-english-v3.0'
    )

def rerank(query, retrievals, n):
  """Reranks the retrieved documents based on the given query."""
  
  # Call the cohere rerank function with the provided parameters
  return co.rerank(
    query = query,          # The search query
    documents = retrievals, # The list of retrieved documents to be reranked
    model = 'rerank-english-v2.0', # The model used for reranking
    # The model rerank-english-v2.0 is likely a specific version of a reranking model provided by a service like Cohere. Reranking models are used to reorder a list of retrieved documents based on their relevance to a given query. This particular model is designed to work with English text and is version 2.0, indicating it is an updated or improved version of a previous model.

    # For more detailed information, you would typically refer to the documentation provided by the service (e.g., Cohere) that offers this model. The documentation would provide insights into the model's architecture, training data, and performance characteristics.
    top_n = n,              # The number of top documents to retrieve
  )