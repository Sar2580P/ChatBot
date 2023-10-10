from langchain.document_loaders import WikipediaLoader
from langchain.text_splitter import CharacterTextSplitter
import os
from langchain.vectorstores.neo4j_vector import Neo4jVector
# Read the wikipedia article
raw_documents = WikipediaLoader(query="The Witcher").load()
# Define chunking strategy
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, chunk_overlap=20
)
# Chunk the document
documents = text_splitter.split_documents(raw_documents)
# Remove the summary
for d in documents:
    del d.metadata["summary"]

#_________________________________________________________________________

# Each text chunk is stored in Neo4j as a single isolated node.

'''
By default, Neo4j vector index implementation in LangChain represents docs using Chunk node label, 
where text property stores text of docs, and the embedding property holds vector representation of text. 
The implementation allows you to customize the node label, text and embedding property names.
'''
neo4j_db = Neo4jVector.from_documents(
    documents,
    Embedding(),
    url=os.environ['NEO4J_URI'],
    username=os.environ['NEO4J_USERNAME'],
    password=os.environ['NEO4J_PASSWORD'],
    database="neo4j",  # neo4j by default
    index_name="wikipedia",  # vector by default
    node_label="WikipediaArticle",  # Chunk by default
    text_node_property="info",  # text by default
    embedding_node_property="vector",  # embedding by default
    create_id_index=True,  # True by default
)
print(neo4j_db.query('Show constraints'))