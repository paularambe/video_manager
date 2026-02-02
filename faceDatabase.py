import psycopg2
from psycopg2.extras import DictCursor
from insightface.app import FaceAnalysis
from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np



@dataclass
class PersonEmbedding:
    person_id: int
    name: str
    embedding: np.ndarray  # shape (512,)



class FaceDatabase:
    """
    Capa sencilla para PostgreSQL:
    - persons (id, name)
    - face_embeddings (id, person_id, embedding BYTEA)
    """

    def __init__(self, host="localhost", port=5432,
                 dbname="espface_db", user="espface",
                 password="espraspi"):
        self.conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password,
            cursor_factory=DictCursor,
        )
        self.conn.autocommit = True
        self._ensure_schema()

    def _ensure_schema(self):
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS persons (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS face_embeddings (
                    id SERIAL PRIMARY KEY,
                    person_id INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
                    embedding BYTEA NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)

    def get_or_create_person(self, name: str) -> int:
        name = name.strip()
        if not name:
            raise ValueError("Nombre vacío")

        with self.conn.cursor() as cur:
            # Buscar si ya existe
            cur.execute("SELECT id FROM persons WHERE name = %s", (name,))
            row = cur.fetchone()
            if row:
                return int(row["id"])

            # Crear
            cur.execute(
                "INSERT INTO persons (name) VALUES (%s) RETURNING id",
                (name,),
            )
            row = cur.fetchone()
            return int(row["id"])

    def add_embedding(self, person_id: int, emb: np.ndarray):
        """
        Guarda embedding (512 float32) como BYTEA.
        """
        emb = np.asarray(emb, dtype=np.float32).flatten()
        if emb.shape[0] != 512:
            raise ValueError(f"Embedding debe ser de 512D, obtenido {emb.shape}")
        emb_bytes = emb.tobytes()

        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO face_embeddings (person_id, embedding) VALUES (%s, %s)",
                (person_id, psycopg2.Binary(emb_bytes)),
            )

    def load_all_embeddings(self) -> List[PersonEmbedding]:
        """
        Devuelve todos los embeddings (para tenerlos en memoria).
        """
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT fe.person_id, p.name, fe.embedding
                FROM face_embeddings fe
                JOIN persons p ON p.id = fe.person_id
            """)
            rows = cur.fetchall()

        result: List[PersonEmbedding] = []
        for r in rows:
            emb_bytes = bytes(r["embedding"])
            emb = np.frombuffer(emb_bytes, dtype=np.float32)
            if emb.shape[0] != 512:
                continue
            # Aseguramos normalización L2
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            result.append(PersonEmbedding(
                person_id=int(r["person_id"]),
                name=str(r["name"]),
                embedding=emb,
            ))
        return result

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass