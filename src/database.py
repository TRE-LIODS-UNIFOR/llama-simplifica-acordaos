import functools
import sqlite3
from datetime import datetime

from models import Document, Prompt, Response
from relatorio import table_to_csv

class PromptDB:
    def __init__(self, db_path='data/prompts.db'):
        self.conn = sqlite3.connect(db_path)
        self.create_prompts_table()
        self.create_responses_table()
        self.create_documents_table()

    def create_prompts_table(self):
        query = """
        CREATE TABLE IF NOT EXISTS prompts (
            id TEXT PRIMARY KEY,
            prompt TEXT NOT NULL,
            temperature REAL NOT NULL,
            top_p REAL NOT NULL,
            top_k REAL NOT NULL,
            context_size INTEGER NOT NULL,
            embeddings_model TEXT NOT NULL,
            chunk_size INTEGER NOT NULL,
            chunk_overlap INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            UNIQUE (prompt, temperature, top_p, top_k, context_size, embeddings_model, chunk_size, chunk_overlap)
        );
        """
        self.conn.execute(query)

    def create_responses_table(self):
        query = """
        CREATE TABLE IF NOT EXISTS responses (
            id TEXT PRIMARY KEY,
            prompt_id TEXT NOT NULL,
            response TEXT NOT NULL,
            quality INTEGER,
            document_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(prompt_id) REFERENCES prompts(id),
            FOREIGN KEY(document_id) REFERENCES documents(id)
        );
        """
        self.conn.execute(query)

    def create_documents_table(self):
        query = """
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY
        );
        """
        self.conn.execute(query)

    def insert_prompt(self, prompt: Prompt):
        query_check = """
        SELECT id FROM prompts
        WHERE prompt = ? AND temperature = ? AND context_size = ? AND top_p = ? AND top_k = ?
        AND embeddings_model = ? AND chunk_size = ? AND chunk_overlap = ?
        """

        query_insert = """
        INSERT INTO prompts (id, prompt, temperature, context_size, top_p, top_k, embeddings_model, chunk_size, chunk_overlap)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        cursor = self.conn.cursor()
        cursor.execute(
            query_check,
            (
                prompt.prompt,
                prompt.temperature,
                prompt.context_size,
                prompt.top_p,
                prompt.top_k,
                prompt.embeddings_model,
                prompt.chunk_size,
                prompt.chunk_overlap,
            )
        )
        result = cursor.fetchone()

        if result:
            print('Existing prompt.')
            return result[0]
        else:
            print('New prompt.')
            cursor.execute(
                query_insert,
                (
                    prompt.id,
                    prompt.prompt,
                    prompt.temperature,
                    prompt.context_size,
                    prompt.top_p,
                    prompt.top_k,
                    prompt.embeddings_model,
                    prompt.chunk_size,
                    prompt.chunk_overlap,
                )
            )
            self.conn.commit()
            cursor.execute("SELECT id FROM prompts WHERE rowid = ?", (cursor.lastrowid,))
            row = cursor.fetchone()
            return row[0]


    def insert_response(self, response: Response):
        query = """
        INSERT INTO responses (id, prompt_id, response, quality, document_id)
        VALUES (?, ?, ?, ?, ?)
        """
        self.conn.execute(
            query,
            (
                response.id,
                response.prompt_id,
                response.response,
                response.quality,
                response.document_id
            )
        )

    def insert_document(self, document: Document) -> str:
        query_check = """
        SELECT id FROM documents
        WHERE id = ?
        """

        query = """
        INSERT INTO documents (id)
        VALUES (?)
        """

        cursor = self.conn.cursor()
        cursor.execute(
            query_check,
            (
                document.id,
            )
        )
        result = cursor.fetchone()

        if result:
            # print('Existing document.')
            return result[0]
        else:
            print('New document.')
            self.conn.execute(
                query,
                (
                    document.id,
                )
            )
            self.conn.commit()
            return document.id

    def fetch_prompt(self, prompt: Prompt):
        query = """
        SELECT * FROM prompts
        WHERE prompts.prompt = (?)
        """
        return self.conn.execute(
            query,
            (
                prompt
            )
        )

    def get_prompts(self):
        query = """
        SELECT * FROM prompts
        """
        cursor = self.conn.execute(query)
        return cursor.fetchall()

    def get_responses(self):
        query = """
        SELECT * FROM responses
        """
        cursor = self.conn.execute(query)
        return cursor.fetchall()

    def get_prompt_responses(self, prompt_id):
        query = """
        SELECT * FROM responses
        WHERE prompt_id LIKE ?
        """
        cursor = self.conn.execute(
            query,
            (
                f'{prompt_id}%',
            )
        )
        return cursor.fetchall()

    def commit(self):
        self.conn.commit()

    def close(self):
        self.conn.close()


if __name__ == '__main__':
    import argparse
    from dataclasses import fields
    from pprint import pprint

    db = PromptDB()

    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('operation')
    argument_parser.add_argument(
        '-p', '--prompt-id',
    )
    argument_parser.add_argument(
        '-e', '--export',
    )

    arg = argument_parser.parse_args()
    # print(arg)

    def result(operation: str, model, *args):
        print(args)
        if args:
            res = getattr(db, operation)(args)
        else:
            res = getattr(db, operation)()
        if arg.export:
            table_to_csv(res, arg.export, [field.name for field in fields(model)])
        return res

    make_result = functools.partial(result, arg.operation)

    match arg.operation:
        case 'get_prompts':
            res = make_result(Prompt)
            pprint(res)
        case 'get_responses':
            res = make_result(Response)
            pprint(res)
        case 'get_prompt_responses':
            prompt_id = arg.prompt_id
            res = make_result(Response, prompt_id)
            pprint(res)
