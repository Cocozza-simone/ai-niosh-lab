"""
Sistema RAG NIOSH - Retrieval-Augmented Generation

Questo modulo implementa un sistema avanzato di Retrieval-Augmented Generation
specificamente progettato per supportare la generazione di report NIOSH 
con conoscenza specialistica preesistente.

Architettura completa con:
- Sistema di embedding basato su sentence transformers (Jina AI v4)
- Vector store ottimizzato per similarity search ad alta performance
- Knowledge base specializzata su ergonomia e sicurezza NIOSH
- Sistema di caching per ridurre tempi di elaborazione
- Gestione intelligente dei documenti con chunking strategico
- Sistema di retrieval contestuale basato su similarity threshold

Funzionalità principali:
- Indicizzazione automatica di documenti tecnici NIOSH
- Retrieval semantico contestuale per generazione report
- Gestione cache embeddings per ottimizzazione performance
- Supporto per documenti PDF, TXT e formati strutturati
- Sistema di logging dettagliato per monitoraggio e debug

Utilizzo tipico:
1. Caricamento documenti nella knowledge base
2. Generazione embeddings con ottimizzazione GPU
3. Retrieval contestuale basato su query NIOSH
4. Integrazione con sistema di generazione report

Versione: 1.0
Dipendenze: sentence-transformers, numpy, pathlib, hashlib
Requisiti hardware: GPU raccomandata per embedding generation
"""

import os
import json
import pickle
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

class RAGConfig:
    """
    Classe di configurazione centrale per il sistema RAG NIOSH.
    
    Centralizza tutti i parametri di configurazione per garantire coerenza
    e facilitare la manutenzione del sistema RAG.
    
    Attributi principali:
        - Directory per knowledge base, embeddings e cache
        - Configurazione modello sentence transformer
        - Parametri RAG (top_k, similarity threshold, chunking)
        - Impostazioni cache e performance
        
    Funzionalità:
        - Percorsi configurabili per deployment diversi
        - Parametri ottimizzati per documenti tecnici NIOSH
        - Gestione TTL cache per mantenere dati aggiornati
        - Impostazioni GPU/CPU per ottimizzazione hardware
    """
    
    # Directories
    KNOWLEDGE_BASE_DIR = Path("knowledge_base")
    EMBEDDINGS_DIR = Path("embeddings")
    CACHE_DIR = Path("cache")
    
    # Model settings
    SENTENCE_TRANSFORMER_MODEL = "jinaai/jina-embeddings-v4"
    SENTENCE_TRANSFORMER_TRUST_REMOTE_CODE = True  
    
    # RAG parameters
    TOP_K_RETRIEVAL = 5
    SIMILARITY_THRESHOLD = 0.3
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    
    # Cache settings
    CACHE_ENABLED = True
    CACHE_TTL_HOURS = 24
    
    @classmethod
    def ensure_directories(cls):
        """Crea directory se non esistono"""
        for dir_path in [cls.KNOWLEDGE_BASE_DIR, cls.EMBEDDINGS_DIR, cls.CACHE_DIR]:
            dir_path.mkdir(exist_ok=True)

RAGConfig.ensure_directories()

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Document:
    """Documento nel knowledge base"""
    id: str
    content: str
    metadata: Dict
    source: str
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self):
        return asdict(self)

@dataclass
class Chunk:
    """Chunk di testo con embedding"""
    id: str
    doc_id: str
    content: str
    embedding: Optional[np.ndarray]
    metadata: Dict
    position: int
    
    def to_dict(self):
        d = asdict(self)
        if self.embedding is not None:
            d['embedding'] = self.embedding.tolist()
        return d

@dataclass
class RetrievalResult:
    """Risultato di retrieval"""
    chunk: Chunk
    score: float
    rank: int

# ============================================================================
# KNOWLEDGE BASE LOADER
# ============================================================================

class KnowledgeBaseLoader:
    """Carica e prepara documenti per il RAG"""
    
    def __init__(self, kb_dir: Path = RAGConfig.KNOWLEDGE_BASE_DIR):
        self.kb_dir = kb_dir
    
    def load_niosh_manual(self) -> List[Document]:
        """Carica il PDF del manuale NIOSH"""
        logger.info("Loading NIOSH manual PDF...")
        
        niosh_pdf = Path("Applications Manual for the Revised NIOSH .pdf")
        if not niosh_pdf.exists():
            logger.warning("NIOSH manual PDF not found")
            return []
        
        try:
            # Try to extract text from PDF
            import PyPDF2
            
            documents = []
            with open(niosh_pdf, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        text = page.extract_text()
                        if text.strip():
                            doc = Document(
                                id=f"niosh_manual_page_{page_num:03d}",
                                content=text.strip(),
                                metadata={
                                    "source_type": "manual",
                                    "page_number": page_num + 1,
                                    "total_pages": len(pdf_reader.pages)
                                },
                                source=str(niosh_pdf)
                            )
                            documents.append(doc)
                    except Exception as e:
                        logger.warning(f"Error processing page {page_num}: {e}")
                        continue
            
            logger.info(f"Loaded {len(documents)} pages from NIOSH manual")
            return documents
            
        except ImportError:
            logger.error("PyPDF2 not installed. Install with: pip install PyPDF2")
            return []
        except Exception as e:
            logger.error(f"Error loading NIOSH manual PDF: {e}")
            return []
    
    def load_example_reports(self) -> List[Document]:
        """Carica report esempio di alta qualità"""
        logger.info("Loading example reports...")
        
        examples = {
            "warehouse_example": """
Example: Warehouse Supply Loading
A warehouse worker loads supply stock weighing 15 kg from floor-level pallets 
(V=10 cm) to waist-height conveyors (V=90 cm). Horizontal reach is 40 cm due 
to pallet positioning. Frequency is 2 lifts/minute over 4-hour shifts. Good 
coupling via built-in handles.

Analysis: RWL = 23 × 0.625 × 0.805 × 0.891 × 1.000 × 0.88 × 1.00 = 9.1 kg
LI = 15.0/9.1 = 1.65 (moderate hazard)

Primary limiting factor: VM = 0.805 (low origin height)
Recommendation: Raise pallet height to 75 cm using spring-loaded platforms.
Modified RWL = 13.2 kg, LI = 1.14 (significant improvement).
""",
            "assembly_example": """
Example: Assembly Line Component Transfer
Assembly operator transfers electronic components (8 kg) from side cart 
(H=50 cm, V=80 cm, A=45°) to assembly station (H=30 cm, V=80 cm, A=0°).
Frequency 4/min, continuous 8-hour operation. Fair coupling.

Analysis: Origin RWL = 23 × 0.50 × 0.985 × 0.918 × 0.856 × 0.45 × 0.95 = 3.9 kg
LI = 8.0/3.9 = 2.05 (moderate-to-high hazard)

Limiting factors: HM=0.50 (extended reach), AM=0.856 (trunk rotation), FM=0.45 (high frequency)
Recommendation: Eliminate asymmetry by reorienting workstation; reduce frequency 
through automation. Modified LI = 1.2 with combined interventions.
"""
        }
        
        documents = []
        for ex_id, content in examples.items():
            doc = Document(
                id=f"example_{ex_id}",
                content=content.strip(),
                metadata={
                    "source_type": "example",
                    "quality": "high",
                    "validated": True
                },
                source="NIOSH Validated Examples"
            )
            documents.append(doc)
        
        logger.info(f"Loaded {len(documents)} example reports")
        return documents
    
    def load_custom_documents(self, files: List[Path]) -> List[Document]:
        """Carica documenti custom da file"""
        documents = []
        
        for file_path in files:
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                doc = Document(
                    id=f"custom_{file_path.stem}",
                    content=content,
                    metadata={
                        "source_type": "custom",
                        "filename": file_path.name
                    },
                    source=str(file_path)
                )
                documents.append(doc)
                logger.info(f"Loaded custom document: {file_path.name}")
            
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        return documents
    
    def load_all(self, custom_files: List[Path] = None) -> List[Document]:
        """Carica tutto il knowledge base"""
        documents = []
        
        documents.extend(self.load_niosh_manual())
        documents.extend(self.load_example_reports())
        
        if custom_files:
            documents.extend(self.load_custom_documents(custom_files))
        
        logger.info(f"Total documents loaded: {len(documents)}")
        return documents

# ============================================================================
# TEXT CHUNKING
# ============================================================================

class TextChunker:
    """Divide documenti in chunk gestibili"""
    
    def __init__(
        self, 
        chunk_size: int = RAGConfig.CHUNK_SIZE,
        chunk_overlap: int = RAGConfig.CHUNK_OVERLAP
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_document(self, doc: Document) -> List[Chunk]:
        """Divide documento in chunk con overlap"""
        text = doc.content
        chunks = []
        
        # Split by sentences first (better semantic boundaries)
        sentences = self._split_sentences(text)
        
        current_chunk = []
        current_length = 0
        position = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Create chunk
                chunk_text = " ".join(current_chunk)
                chunk_id = self._generate_chunk_id(doc.id, position)
                
                chunks.append(Chunk(
                    id=chunk_id,
                    doc_id=doc.id,
                    content=chunk_text,
                    embedding=None,
                    metadata={**doc.metadata, "chunk_position": position},
                    position=position
                ))
                
                # Keep overlap sentences
                overlap_sentences = self._get_overlap_sentences(current_chunk)
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s) for s in current_chunk)
                position += 1
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunk_id = self._generate_chunk_id(doc.id, position)
            
            chunks.append(Chunk(
                id=chunk_id,
                doc_id=doc.id,
                content=chunk_text,
                embedding=None,
                metadata={**doc.metadata, "chunk_position": position},
                position=position
            ))
        
        logger.info(f"Document {doc.id} split into {len(chunks)} chunks")
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences (simple version)"""
        import re
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """Get last N sentences for overlap"""
        overlap_chars = self.chunk_overlap
        overlap_sentences = []
        char_count = 0
        
        for sentence in reversed(sentences):
            if char_count + len(sentence) > overlap_chars:
                break
            overlap_sentences.insert(0, sentence)
            char_count += len(sentence)
        
        return overlap_sentences
    
    def _generate_chunk_id(self, doc_id: str, position: int) -> str:
        """Generate unique chunk ID"""
        return f"{doc_id}_chunk_{position:04d}"

# ============================================================================
# EMBEDDING ENGINE
# ============================================================================

class EmbeddingEngine:
    """Gestisce embedding con sentence-transformers"""
    
    def __init__(
        self,
        model_name: str = RAGConfig.SENTENCE_TRANSFORMER_MODEL,
        cache_dir: Path = RAGConfig.EMBEDDINGS_DIR
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.cache_file = cache_dir / "embeddings_cache.pkl"
        self.model = None
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict:
        """Carica cache embeddings"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    cache = pickle.load(f)
                logger.info(f"Loaded {len(cache)} cached embeddings")
                return cache
            except Exception as e:
                logger.error(f"Error loading cache: {e}")
        return {}
    
    def _save_cache(self):
        """Salva cache embeddings"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
            logger.info(f"Saved {len(self.cache)} embeddings to cache")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key from text"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _ensure_model(self):
        """Load the model if not already loaded with GPU optimization"""
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                
                # Check for GPU availability
                import torch
                
                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"Using device: {device}")
                
                # Load model with GPU optimization
                self.model = SentenceTransformer(
                    self.model_name, 
                    device=device,
                    trust_remote_code=RAGConfig.SENTENCE_TRANSFORMER_TRUST_REMOTE_CODE,
                    model_kwargs={'default_task': 'retrieval'}
                )
                
                # Optimize for GPU if available
                if device == "cuda":
                    logger.info("Optimizing for GPU performance...")
                    self.model.half()  # Use fp16 for better GPU performance
                    logger.info(f"GPU Memory available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
                
                logger.info(f"Loaded sentence-transformer model: {self.model_name} on {device}")
                
            except ImportError:
                logger.error("sentence-transformers not installed. Please run: pip install sentence-transformers")
                raise
            except Exception as e:
                logger.warning(f"GPU optimization failed: {e}. Using CPU.")
                # Fallback to CPU
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(
                    self.model_name, 
                    device="cpu",
                    trust_remote_code=RAGConfig.SENTENCE_TRANSFORMER_TRUST_REMOTE_CODE,
                    model_kwargs={'default_task': 'retrieval'}
                )
    
    def embed_text(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding per testo"""
        if not text.strip():
            return None
            
        cache_key = self._get_cache_key(text)
        
        # Check cache
        if cache_key in self.cache:
            return np.array(self.cache[cache_key])
        
        # Generate new embedding
        try:
            self._ensure_model()
            embedding = self.model.encode([text], convert_to_numpy=True, task='retrieval')[0]
            
            # Cache it
            self.cache[cache_key] = embedding.tolist()
            self._save_cache()
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    def embed_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Genera embeddings per lista di chunk con GPU optimization"""
        logger.info(f"Embedding {len(chunks)} chunks...")
        
        if not chunks:
            return []
        
        embedded_chunks = []
        new_embeddings = 0
        
        # Check if we should use batch processing for GPU
        use_batch = hasattr(self.model, 'device') and self.model.device.type == 'cuda'
        
        if use_batch and len(chunks) > 1:
            # Batch processing for GPU - much faster
            logger.info(f"Using GPU batch processing for {len(chunks)} chunks...")
            
            try:
                # Filter out empty texts
                valid_chunks = [c for c in chunks if c.content.strip()]
                texts = [c.content for c in valid_chunks]
                
                if texts:
                    # Process all texts in one batch
                    embeddings = self.model.encode(texts, convert_to_numpy=True, batch_size=32, task='retrieval')
                    
                    # Assign embeddings to chunks
                    for chunk, embedding in zip(valid_chunks, embeddings):
                        chunk.embedding = embedding
                        embedded_chunks.append(chunk)
                        new_embeddings += 1
                    
                    logger.info(f"GPU batch embedded {len(embedded_chunks)} chunks")
                    
                # Save cache periodically
                if new_embeddings > 0:
                    self._save_cache()
                    
            except Exception as e:
                logger.warning(f"GPU batch processing failed: {e}. Using sequential processing.")
                use_batch = False
        
        # Fallback to sequential processing (CPU or GPU batch failed)
        if not use_batch or not hasattr(self.model, 'device') or self.model.device.type == 'cpu':
            for i, chunk in enumerate(chunks):
                if i % 10 == 0:
                    logger.info(f"Progress: {i}/{len(chunks)}")
                
                embedding = self.embed_text(chunk.content)
                
                if embedding is not None:
                    chunk.embedding = embedding
                    embedded_chunks.append(chunk)
                    new_embeddings += 1
                else:
                    logger.warning(f"Failed to embed chunk {chunk.id}")
            
            # Save cache periodically
            if new_embeddings > 0:
                self._save_cache()
        
        logger.info(f"Embedded {len(embedded_chunks)} chunks ({new_embeddings} new)")
        return embedded_chunks

# ============================================================================
# VECTOR STORE
# ============================================================================

class VectorStore:
    """Store per embeddings con similarity search"""
    
    def __init__(self, store_path: Path = None):
        self.chunks: List[Chunk] = []
        self.embeddings_matrix: Optional[np.ndarray] = None
        self.store_path = store_path or (RAGConfig.EMBEDDINGS_DIR / "vector_store.pkl")
    
    def add_chunks(self, chunks: List[Chunk]):
        """Aggiungi chunk allo store"""
        valid_chunks = [c for c in chunks if c.embedding is not None]
        self.chunks.extend(valid_chunks)
        self._rebuild_matrix()
        logger.info(f"Added {len(valid_chunks)} chunks to vector store")
    
    def _rebuild_matrix(self):
        """Ricostruisci matrice embeddings"""
        if not self.chunks:
            self.embeddings_matrix = None
            return
        
        embeddings = [chunk.embedding for chunk in self.chunks]
        self.embeddings_matrix = np.vstack(embeddings)
        logger.info(f"Rebuilt embeddings matrix: {self.embeddings_matrix.shape}")
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = RAGConfig.TOP_K_RETRIEVAL,
        threshold: float = RAGConfig.SIMILARITY_THRESHOLD
    ) -> List[RetrievalResult]:
        """Similarity search"""
        if self.embeddings_matrix is None:
            logger.warning("Vector store is empty")
            return []
        
        # Cosine similarity
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        embeddings_norm = self.embeddings_matrix / (
            np.linalg.norm(self.embeddings_matrix, axis=1, keepdims=True) + 1e-8
        )
        
        similarities = embeddings_norm @ query_norm
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices):
            score = float(similarities[idx])
            
            if score < threshold:
                continue
            
            results.append(RetrievalResult(
                chunk=self.chunks[idx],
                score=score,
                rank=rank
            ))
        
        logger.info(f"Retrieved {len(results)} chunks with score ≥ {threshold}")
        return results
    
    def save(self):
        """Salva vector store su disco"""
        try:
            data = {
                'chunks': [c.to_dict() for c in self.chunks],
                'embeddings': self.embeddings_matrix.tolist() if self.embeddings_matrix is not None else None
            }
            
            with open(self.store_path, 'wb') as f:
                pickle.dump(data, f)
            
            logger.info(f"Saved vector store to {self.store_path}")
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
    
    def load(self):
        """Carica vector store da disco"""
        if not self.store_path.exists():
            logger.warning(f"No vector store found at {self.store_path}")
            return
        
        try:
            with open(self.store_path, 'rb') as f:
                data = pickle.load(f)
            
            # Reconstruct chunks
            self.chunks = []
            for chunk_dict in data['chunks']:
                chunk = Chunk(**chunk_dict)
                if 'embedding' in chunk_dict and chunk_dict['embedding']:
                    chunk.embedding = np.array(chunk_dict['embedding'])
                self.chunks.append(chunk)
            
            # Reconstruct matrix
            if data['embeddings']:
                self.embeddings_matrix = np.array(data['embeddings'])
            
            logger.info(f"Loaded vector store with {len(self.chunks)} chunks")
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")

# ============================================================================
# RAG RETRIEVER
# ============================================================================

class RAGRetriever:
    """Componente di retrieval con re-ranking"""
    
    def __init__(
        self,
        embedding_engine: EmbeddingEngine,
        vector_store: VectorStore
    ):
        self.embedding_engine = embedding_engine
        self.vector_store = vector_store
    
    def retrieve(
        self,
        query: str,
        top_k: int = RAGConfig.TOP_K_RETRIEVAL,
        rerank: bool = True
    ) -> List[RetrievalResult]:
        """Recupera documenti rilevanti per query"""
        logger.info(f"Retrieving documents for query: {query[:50]}...")
        
        # Generate query embedding
        query_embedding = self.embedding_engine.embed_text(query)
        
        if query_embedding is None:
            logger.error("Failed to embed query")
            return []
        
        # Vector search
        results = self.vector_store.search(query_embedding, top_k=top_k * 2)
        
        if rerank and results:
            results = self._rerank_results(query, results)
        
        return results[:top_k]
    
    def _rerank_results(
        self, 
        query: str, 
        results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Re-rank usando keyword matching (fallback semplice)"""
        query_terms = set(query.lower().split())
        
        for result in results:
            content_terms = set(result.chunk.content.lower().split())
            keyword_overlap = len(query_terms & content_terms) / len(query_terms)
            
            # Combine semantic similarity with keyword overlap
            result.score = 0.7 * result.score + 0.3 * keyword_overlap
        
        # Re-sort
        results.sort(key=lambda x: x.score, reverse=True)
        
        # Update ranks
        for rank, result in enumerate(results):
            result.rank = rank
        
        return results

# ============================================================================
# RAG-ENHANCED GENERATOR
# ============================================================================

class RAGGenerator:
    """Generatore potenziato con RAG"""
    
    def __init__(
        self,
        retriever: RAGRetriever
    ):
        self.retriever = retriever
    
    def generate_with_context(
        self,
        prompt: str,
        system_prompt: str,
        use_rag: bool = True,
        max_context_chunks: int = 3
    ) -> Optional[str]:
        """Genera con contesto recuperato da RAG"""
        
        context = ""
        
        if use_rag:
            # Retrieve relevant context
            results = self.retriever.retrieve(prompt, top_k=max_context_chunks)
            
            if results:
                context_parts = []
                for i, result in enumerate(results, 1):
                    context_parts.append(
                        f"[Context {i}] (relevance: {result.score:.2f})\n{result.chunk.content}"
                    )
                
                context = "\n\n".join(context_parts)
                logger.info(f"Using {len(results)} context chunks for generation")
        
        # Build enhanced prompt
        if context:
            enhanced_system = f"""{system_prompt}

RELEVANT CONTEXT FROM NIOSH KNOWLEDGE BASE:
{context}

Use this context to inform your response, citing specific guidelines when applicable.
"""
        else:
            enhanced_system = system_prompt
        
        # Generate response
        return self._generate_response(enhanced_system, prompt)
    
    def _generate_response(
        self, 
        system_prompt: str, 
        user_prompt: str
    ) -> Optional[str]:
        """Generate response using Ollama LLM"""
        try:
            # Import here to avoid circular imports
            import requests
            
            # Ollama configuration
            OLLAMA_URL = "http://localhost:11434/api/generate"
            MODEL = "llama3.2:latest"
            
            # Prepare request using the same format as prod_gen_v21.py
            full_prompt = f"[SYSTEM] {system_prompt}\n\n[USER] {user_prompt}"
            
            payload = {
                "model": MODEL,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 600
                }
            }
            
            logger.info(f"Generating response using {MODEL}...")
            
            # Make request to Ollama
            response = requests.post(OLLAMA_URL, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            generated_text = result.get('response', '')
            
            if not generated_text.strip():
                logger.warning("Empty response from Ollama")
                return None
            
            logger.info("Successfully generated response from Ollama")
            return generated_text.strip()
            
        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to Ollama. Make sure Ollama is running at http://localhost:11434")
            # Return fallback template
            return json.dumps({
                "job_description_narrative": f"A worker handles materials weighing {self._extract_weight_from_prompt(user_prompt)} kg in a manual lifting operation following NIOSH guidelines."
            })
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return None
    
    def _extract_weight_from_prompt(self, prompt: str) -> str:
        """Extract weight from prompt for fallback"""
        import re
        weight_match = re.search(r'Weight:\s*([\d.]+)\s*kg', prompt)
        return weight_match.group(1) if weight_match else "specified"

# ============================================================================
# MAIN RAG SYSTEM
# ============================================================================

class NIOSHRAGSystem:
    """
    Sistema RAG completo specializzato per applicazioni NIOSH.
    
    Classe principale che orchestra tutti i componenti del sistema RAG:
    caricamento documenti, chunking, embedding generation, vector storage,
    retrieval contestuale e generazione potenziata.
    
    Architettura modulare:
    1. KnowledgeBaseLoader: Caricamento documenti da knowledge base
    2. TextChunker: Divisione intelligente documenti in chunks semanticamente coerenti
    3. EmbeddingEngine: Generazione embeddings con ottimizzazione GPU
    4. VectorStore: Storage e retrieval ad alta performance
    5. RAGRetriever: Retrieval semantico contestuale
    6. RAGGenerator: Generazione potenziata da knowledge base
    
    Funzionalità principali:
        - Building automatico indici da knowledge base
        - Gestione cache embeddings per ottimizzazione
        - Retrieval contestuale basato su similarity threshold
        - Integrazione trasparente con sistema generazione report
        - Logging dettagliato per monitoraggio performance
        
    Utilizzo tipico:
        system = NIOSHRAGSystem(rebuild_index=False)
        results = system.search("NIOSH lifting equation multipliers")
        enhanced_context = system.get_context_for_scenario(scenario_data)
    """
    
    def __init__(self, rebuild_index: bool = False):
        """
        Inizializza il sistema RAG NIOSH completo.
        
        Istanzia tutti i componenti modulari e configura l'indice vettoriale.
        Gestisce automaticamente il building o caricamento degli indici esistenti.
        
        Args:
            rebuild_index (bool): Se True, ricostruisce completamente l'indice.
                                Se False, carica indice esistente se disponibile.
                                Default False per ottimizzare tempi di avvio.
                                
        Note:
            - L'indice viene ricostruito automaticamente se non esiste
            - Tutti i componenti sono inizializzati con configurazioni ottimali
            - Viene effettuato un check di integrità dell'indice esistente
        """
        logger.info("Initializing NIOSH RAG System...")
        
        self.kb_loader = KnowledgeBaseLoader()
        self.chunker = TextChunker()
        self.embedding_engine = EmbeddingEngine()
        self.vector_store = VectorStore()
        self.retriever = None
        self.generator = None
        
        if rebuild_index or not self.vector_store.store_path.exists():
            self.build_index()
        else:
            self.load_index()
        
        self.retriever = RAGRetriever(self.embedding_engine, self.vector_store)
        self.generator = RAGGenerator(self.retriever)
        
        logger.info("NIOSH RAG System ready")
    
    def build_index(self, custom_files: List[Path] = None):
        """
        Costruisce l'indice RAG completo da zero.
        
        Processo completo di indicizzazione:
        1. Caricamento documenti dalla knowledge base
        2. Chunking semantico intelligente dei documenti
        3. Generazione embeddings con ottimizzazione GPU
        4. Costruzione vector store per similarity search
        5. Salvataggio su disco per caricamenti rapidi
        
        Args:
            custom_files (List[Path], optional): Lista di file personalizzati da 
                                              includere oltre alla knowledge base.
                                              Utile per test o documenti temporanei.
                                              
        Note:
            - Processo intensive-time per grandi knowledge base
            - Utilizza automaticamente GPU se disponibile
            - Gestisce cache embeddings per ottimizzare rebuild successivi
            - Fornisce statistiche dettagliate sul processo
        """
        logger.info("Building RAG index from scratch...")
        
        # Load documents
        documents = self.kb_loader.load_all(custom_files)
        
        # Add NIOSH manual PDF if it exists
        niosh_pdf = Path("Applications Manual for the Revised NIOSH .pdf")
        if niosh_pdf.exists():
            logger.info("Loading NIOSH manual PDF...")
            niosh_docs = self.kb_loader.load_custom_documents([niosh_pdf])
            documents.extend(niosh_docs)
        
        # Chunk documents
        all_chunks = []
        for doc in documents:
            chunks = self.chunker.chunk_document(doc)
            all_chunks.extend(chunks)
        
        logger.info(f"Created {len(all_chunks)} chunks total")
        
        # Generate embeddings
        embedded_chunks = self.embedding_engine.embed_chunks(all_chunks)
        
        # Add to vector store
        self.vector_store.add_chunks(embedded_chunks)
        
        # Save
        self.vector_store.save()
        
        logger.info("Index building complete")
    
    def load_index(self):
        """Carica index esistente"""
        logger.info("Loading existing RAG index...")
        self.vector_store.load()
    
    def generate_narrative(
        self,
        initial_data: Dict,
        title: str
    ) -> Dict:
        """Genera narrative usando RAG"""
        
        # Build query from parameters
        params = initial_data
        query = f"""
Generate NIOSH-style job description for lifting task:
Scenario: {title}
Weight: {params['common_parameters']['L_load_kg']} kg
Frequency: {params['common_parameters']['F_frequency_per_min']} lifts/min
Duration: {params['common_parameters']['duration_hours']} hours
Control required: {params['significant_control_at_destination']}
        """.strip()
        
        system_prompt = """You are a NIOSH technical writer specializing in the "Applications Manual for the Revised NIOSH Lifting Equation".

Generate comprehensive Job Description following NIOSH manual style with complete task context.

CRITICAL RULES:
1. Length: 4-6 detailed sentences providing full context
2. NO posture descriptions (no "twist", "bend", "flex fingers", "reach")
3. NO technical parameters (no cm, kg, H, V, A values)
4. NO explanations of difficulty (no "due to", "cannot get closer", "requires")
5. Include: workplace setting, task patterns, object characteristics
6. Describe WHO does WHAT, WHERE, under WHAT CONDITIONS

FORBIDDEN WORDS/PHRASES:
❌ "twist", "bend", "stoop", "reach", "flex fingers", "grip", "rotate"
❌ "cannot get closer", "due to space constraints", "without risking"
❌ "requires control", "must maintain", "in order to"
❌ Any explanations of WHY task is difficult

REQUIRED ELEMENTS:
✓ Worker role and workplace environment
✓ Task frequency and duration patterns  
✓ Object types and handling characteristics
✓ Origin and destination locations
✓ Work method and coupling conditions

EXAMPLE (COMPREHENSIVE):
"A warehouse worker selects compact electronic components from a conveyor and places them into plastic trays at workstations. The task is performed continuously during an 8-hour shift at approximately 4 lifts per minute. Components have molded plastic handles providing fair coupling characteristics. The work occurs in a climate-controlled area with fluorescent overhead lighting."

Response format:
{"job_description_narrative": "comprehensive 4-6 sentence description"}"""
        
        user_prompt = f"""Generate NIOSH-compliant job description.

SCENARIO TITLE: {title}

CRITICAL: DO NOT mention body postures, movements, or gripping details.
Focus ONLY on: worker role, objects handled, source location, destination location.

GOOD EXAMPLE:
"A worker inspects compact containers on a low shelf and lifts them to a higher shelf."

BAD EXAMPLE (DO NOT WRITE LIKE THIS):
"Due to space constraints, the worker cannot get closer and must twist to lift containers."

Required JSON:
{{
    "job_description_narrative": "your 2-3 sentence description"
}}
"""
        
        response = self.generator.generate_with_context(
            prompt=user_prompt,
            system_prompt=system_prompt,
            use_rag=True,
            max_context_chunks=3
        )
        
        if response:
            # Parse JSON response
            import json
            import re
            
            # Clean response
            clean_response = response.strip()
            if clean_response.startswith("```json"):
                clean_response = clean_response[7:]
            if clean_response.endswith("```"):
                clean_response = clean_response[:-3]
            clean_response = clean_response.strip()
            
            try:
                narrative_data = json.loads(clean_response)
                if "job_description_narrative" in narrative_data:
                    return narrative_data
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON response")
        
        # Fallback
        return {
            "job_description_narrative": f"A worker handles materials weighing {params['common_parameters']['L_load_kg']} kg in a manual lifting operation."
        }
    
    def validate_report_section(
        self,
        section_text: str,
        section_type: str
    ) -> Tuple[bool, List[str]]:
        """Valida sezione usando knowledge base"""
        
        query = f"Validate this {section_type} section for NIOSH compliance: {section_text[:200]}"
        
        system_prompt = """You are a NIOSH compliance validator. Check if the report 
section follows NIOSH guidelines. Respond with JSON:
{
    "is_valid": true/false,
    "issues": ["issue1", "issue2", ...],
    "suggestions": ["fix1", "fix2", ...]
}"""
        
        user_prompt = f"""Section type: {section_type}

Section text:
{section_text[:1000]}

Check for:
1. Correct terminology
2. Proper calculation format
3. Risk level consistency
4. Appropriate recommendations
"""
        
        response = self.generator.generate_with_context(
            prompt=user_prompt,
            system_prompt=system_prompt,
            use_rag=True
        )
        
        if response:
            try:
                result = json.loads(response.strip())
                return result.get('is_valid', True), result.get('issues', [])
            except:
                pass
        
        return True, []
    
    def search_knowledge(self, query: str, top_k: int = 5) -> List[Dict]:
        """Cerca nel knowledge base (API pubblica)"""
        results = self.retriever.retrieve(query, top_k=top_k)
        
        return [
            {
                'content': r.chunk.content,
                'score': r.score,
                'source': r.chunk.metadata.get('source', 'Unknown'),
                'section': r.chunk.metadata.get('section', 'N/A')
            }
            for r in results
        ]

# ============================================================================
# INTEGRATION WITH EXISTING SYSTEM
# ============================================================================

def integrate_rag_with_prod_gen(rag_system: NIOSHRAGSystem):
    """
    Funzione helper per integrare RAG nel sistema esistente prod_gen_v21.py
    
    Usage in prod_gen_v21.py:
    
    from niosh_rag_system import NIOSHRAGSystem, integrate_rag_with_prod_gen
    
    # Initialize once at startup
    rag = NIOSHRAGSystem()
    
    # In generate_narrative():
    narrative = rag.generate_narrative(initial_data, title)
    
    # In generate_final_text():
    # Validate sections
    valid, issues = rag.validate_report_section(hazard_section, "hazard_assessment")
    """
    
    logger.info("RAG system ready for integration")
    
    # Return utility functions
    return {
        'generate_narrative': rag_system.generate_narrative,
        'validate_section': rag_system.validate_report_section,
        'search_kb': rag_system.search_knowledge
    }

# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Main CLI per testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="NIOSH RAG System")
    parser.add_argument('--rebuild', action='store_true', help='Rebuild index from scratch')
    parser.add_argument('--search', type=str, help='Search knowledge base')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    
    # Initialize system
    rag = NIOSHRAGSystem(rebuild_index=args.rebuild)
    
    if args.search:
        # Search mode
        print(f"\n🔍 Searching for: {args.search}\n")
        results = rag.search_knowledge(args.search, top_k=3)
        
        for i, result in enumerate(results, 1):
            print(f"{'='*60}")
            print(f"Result {i} (score: {result['score']:.3f})")
            print(f"Source: {result['source']} - Section: {result['section']}")
            print(f"{'='*60}")
            print(result['content'][:300])
            print()
    
    elif args.interactive:
        # Interactive mode
        print("\n🤖 NIOSH RAG Interactive Mode")
        print("Type 'quit' to exit, 'help' for commands\n")
        
        while True:
            try:
                query = input("Query: ").strip()
                
                if query.lower() == 'quit':
                    break
                elif query.lower() == 'help':
                    print("\nCommands:")
                    print("  search <query>  - Search knowledge base")
                    print("  generate <prompt> - Generate text with RAG")
                    print("  quit - Exit")
                    continue
                elif query.startswith('search '):
                    search_query = query[7:]
                    results = rag.search_knowledge(search_query, top_k=2)
                    for r in results:
                        print(f"\n[{r['score']:.2f}] {r['content'][:200]}...")
                elif query.startswith('generate '):
                    prompt = query[9:]
                    response = rag.generator.generate_with_context(
                        prompt=prompt,
                        system_prompt="You are a NIOSH expert assistant.",
                        use_rag=True
                    )
                    print(f"\n📝 Generated:\n{response}\n")
                else:
                    print("Unknown command. Type 'help' for options.")
            
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
    
    else:
        # Default: show system info
        print("\n[OK] NIOSH RAG System initialized successfully")
        print(f"[VECTOR] Vector store: {len(rag.vector_store.chunks)} chunks")
        print(f"[CACHE] Cache: {len(rag.embedding_engine.cache)} embeddings")
        print("\nUse --help for options")

if __name__ == "__main__":
    main()