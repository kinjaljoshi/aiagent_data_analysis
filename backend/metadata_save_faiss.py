from google.cloud import bigquery
import json
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

def initialize_bigquery_client(project_id):
    """Initialize and return a BigQuery client."""
    return bigquery.Client(project=project_id)

def fetch_table_names(client, project_id, dataset_id):
    """Fetch all table names from the given dataset."""
    query = f"""
        SELECT table_name
        FROM `{project_id}.{dataset_id}.INFORMATION_SCHEMA.TABLES`
    """
    tables_result = client.query(query).result()
    return [row.table_name for row in tables_result]

def fetch_table_metadata(client, project_id, dataset_id, table_names):
    """Fetch metadata for all tables in the dataset."""
    table_definitions = {"tables": []}
    for table_name in table_names:
        query = f"""
            SELECT
                table_name,
                column_name,
                IFNULL(description, 'No Description') AS column_description
            FROM
                `{project_id}.{dataset_id}.INFORMATION_SCHEMA.COLUMN_FIELD_PATHS`
            WHERE
                table_name = '{table_name}'
        """
        query_results = client.query(query).result()

        table_dict = {
            "table_name": table_name,
            "table_description": f"Metadata for table {table_name}.",
            "columns": []
        }

        for row in query_results:
            table_dict["columns"].append({
                "column_name": row.column_name,
                "column_description": row.column_description
            })

        table_definitions["tables"].append(table_dict)
    
    return table_definitions

def create_documents(table_definitions):
    """Convert table definitions into LangChain Document objects."""
    documents = []
    for table in table_definitions["tables"]:
        table_text = f"Table: {table['table_name']}\nDescription: {table['table_description']}\n\nColumns:\n"
        for column in table["columns"]:
            table_text += f"  - {column['column_name']}: {column['column_description']}\n"
        
        doc = Document(
            page_content=table_text,
            metadata={"table_name": table["table_name"]}
        )
        documents.append(doc)
    
    return documents

def store_in_faiss(documents, model_name="all-MiniLM-L6-v2", index_path="faiss_table_index"):
    """Store documents in FAISS using LangChain's FAISS Wrapper."""
    embedding_model = SentenceTransformerEmbeddings(model_name=model_name)
    vector_db = FAISS.from_documents(documents, embedding_model)
    vector_db.save_local(index_path)
    print(f"Stored {len(documents)} table definitions in FAISS as single documents.")

def main_block():
    project_id = "llm-text-to-sql-445914"
    dataset_id = "llm_text_to_sql"
    
    client = initialize_bigquery_client(project_id)
    table_names = fetch_table_names(client, project_id, dataset_id)
    table_definitions = fetch_table_metadata(client, project_id, dataset_id, table_names)
    documents = create_documents(table_definitions)
    store_in_faiss(documents)

if __name__ == "__main__":
    main_block()
