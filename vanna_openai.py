import os
from vanna.openai import OpenAI_Chat
from vanna.chromadb import ChromaDB_VectorStore
import sqlite3
import pandas as pd
import threading
import json
from openai import OpenAI
import streamlit as st

class CustomVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        config = config or {}
        self.allow_llm_to_see_data = config.get('allow_llm_to_see_data', False)
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)
        self.db_path = None
        self._local = threading.local()
        
        # Get API key from Streamlit secrets or config
        api_key = st.secrets.get("OPENAI_API_KEY", "") if hasattr(st, "secrets") else config.get('api_key', "")
        if not api_key:
            raise ValueError("OpenAI API key not found in Streamlit secrets or config")
        
        self.openai_client = OpenAI(api_key=api_key)

    def connect_to_sqlite(self, db_path):
        """Connect to SQLite database"""
        self.db_path = db_path
        return self._get_connection() is not None

    def _get_connection(self):
        """Get or create a thread-local database connection"""
        if not hasattr(self._local, 'connection'):
            try:
                self._local.connection = sqlite3.connect(self.db_path)
                print(f"Successfully connected to SQLite database at {self.db_path}")
            except Exception as e:
                print(f"Error connecting to database: {str(e)}")
                return None
        return self._local.connection

    def run_sql(self, sql):
        """Execute SQL query and return results as DataFrame"""
        conn = self._get_connection()
        if not conn:
            raise Exception("Database connection not established. Call connect_to_sqlite first.")
        try:
            return pd.read_sql_query(sql, conn)
        except Exception as e:
            print(f"Error executing SQL: {str(e)}")
            return None

    def train_on_schema(self):
        """Train on the database schema"""
        conn = self._get_connection()
        if not conn:
            raise Exception("Database connection not established. Call connect_to_sqlite first.")
        
        # Get all table schemas
        schema_query = """
        SELECT name, sql 
        FROM sqlite_master 
        WHERE type='table' AND sql IS NOT NULL;
        """
        try:
            schemas = pd.read_sql_query(schema_query, conn)
            
            for _, row in schemas.iterrows():
                self.train(ddl=row['sql'])
                # Also train on table documentation
                self.train(documentation=f"Table {row['name']} contains data about {row['name'].lower()}")
            return True
        except Exception as e:
            print(f"Error training on schema: {str(e)}")
            return False

    def close(self):
        """Close database connection"""
        if hasattr(self._local, 'connection'):
            self._local.connection.close()
            del self._local.connection 

    def generate_insights(self, question, sql, df, conversation_history=None):
        """Generate insights from the query results using OpenAI"""
        print("Generating insights...")  # Debug print
        
        if df is None or df.empty:
            print("DataFrame is None or empty")  # Debug print
            return None

        try:
            # Convert DataFrame to a more readable format
            data_description = df.head(10).to_string() if len(df) > 10 else df.to_string()
            data_stats = df.describe().to_string() if df.select_dtypes(include=['number']).columns.any() else ""
            
            # Prepare column information
            columns_info = "\nColumns and their types:\n" + "\n".join([f"- {col}: {df[col].dtype}" for col in df.columns])

            # Format conversation history for context
            context = ""
            if conversation_history:
                relevant_history = [
                    msg for msg in conversation_history[-5:]  # Last 5 messages
                    if msg["role"] in ["user", "assistant"] and "type" not in msg
                ]
                if relevant_history:
                    context = "\nConversation context:\n" + "\n".join(
                        f"{msg['role']}: {msg['content']}" for msg in relevant_history
                    )

            # Prepare the prompt for OpenAI
            prompt = f"""As a data analyst, analyze the following data and provide insights:

Original Question: {question}

{context}

SQL Query Used:
{sql}

{columns_info}

Data Sample:
{data_description}

Statistical Summary:
{data_stats}

Please provide:
1. Key findings and patterns in the data
2. Notable trends or anomalies
3. Business implications or recommendations
4. Any potential areas for further investigation
5. Relevant insights based on the conversation context

Format the response in a clear, bulleted structure."""

            print("Sending request to OpenAI...")  # Debug print

            # Use OpenAI to generate insights
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",  # or your preferred model
                messages=[
                    {"role": "system", "content": "You are a skilled data analyst providing insights from data."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            print("Received response from OpenAI")  # Debug print
            
            if response and response.choices:
                return response.choices[0].message.content
            return None

        except Exception as e:
            print(f"Error in generate_insights: {str(e)}")  # Debug print
            import traceback
            print(traceback.format_exc())  # Print full traceback
            return None

    def generate_sql(self, question, conversation_history=None):
        """Override generate_sql to handle conversation history"""
        if not self.allow_llm_to_see_data:
            print("Warning: LLM is not allowed to see database data. Set allow_llm_to_see_data=True in config if needed.")
        
        # Format conversation history for context
        context = ""
        if conversation_history:
            relevant_history = [
                msg for msg in conversation_history[-5:]  # Last 5 messages
                if msg["role"] in ["user", "assistant"] and "type" not in msg
            ]
            if relevant_history:
                context = "\nPrevious conversation:\n" + "\n".join(
                    f"{msg['role']}: {msg['content']}" for msg in relevant_history
                )
        
        # Add context to the question
        question_with_context = f"{question}{context}"
        return super().generate_sql(question_with_context) 