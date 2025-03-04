import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langchain_community.vectorstores import FAISS
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

# Ensure NLTK resources are downloaded
nltk.download("punkt")
nltk.download("stopwords")

# Load Sentence Transformer Model
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def extract_keywords(query_text):
    """
    Extracts important keywords from a query by removing stopwords and duplicates.
    
    :param query_text: Input search query
    :return: List of extracted keywords
    """
    # Define custom stop words (action verbs & common words)
    custom_stopwords = set(stopwords.words("english")).union(
        {"get", "put", "fetch", "retrieve", "show", "find"}
    )

    # Tokenize & Convert to Lowercase
    words = word_tokenize(query_text.lower())

    # Remove Stop Words & Duplicates (Preserve Order)
    filtered_words = []
    for word in words:
        if word not in custom_stopwords and word not in filtered_words:
            filtered_words.append(word)

    return filtered_words

def search_faiss_index(filtered_keywords, faiss_index_path="faiss_table_index"):
    """
    Searches the FAISS index for each keyword and retrieves relevant documents.

    :param filtered_keywords: List of extracted keywords
    :param faiss_index_path: Path to the FAISS index
    :return: List of retrieved documents
    """
    try:
        # Load FAISS Index
        vector_db = FAISS.load_local(faiss_index_path, embedding_model, allow_dangerous_deserialization=True)
        print(" FAISS index loaded successfully.")

        final_docs = []
        for search_text in filtered_keywords:
            retrieved_docs = vector_db.similarity_search(search_text, k=1)  # Retrieve top-1 match
            if retrieved_docs:
                final_docs.append(retrieved_docs[0])

        return final_docs
    except Exception as e:
        print(f" Error loading FAISS index: {e}")
        return []

def get_query_context(query_text):
    """Extracts keywords and retrieves relevant FAISS context."""
    keywords = extract_keywords(query_text)
    docs = search_faiss_index(keywords)
    return "query_text : ".join([doc.page_content for doc in docs]) if docs else "No relevant context found"


def display_results(results):
    """
    Prints the search results.

    :param results: List of retrieved FAISS documents
    """
    if not results:
        print("No matching results found.")
        return

    print("\n**Search Results:**")
    for idx, doc in enumerate(results, 1):
        print(f"{idx}. {doc.page_content}\n")
