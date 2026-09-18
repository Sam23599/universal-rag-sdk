"""Provider-neutral, modular RAG SDK."""

__version__ = "0.1.0"

from .cache import AsyncTTLCache
from .config import IngestionConfig, PipelineConfig, RetrievalConfig
from .context import ContextBuilder
from .defaults import CallableEmbeddingProvider, CallableLLMProvider, LatestTurnQueryPlanner, RegexPIIRedactor
from .evaluation import Evaluator, InMemoryFeedbackStore
from .facade import RAG
from .indexing import FileSystemDocumentStore, InMemoryDocumentStore, InMemoryVersionedIndex
from .ingestion import IngestionPipeline
from .models import (
    Chunk,
    Citation,
    Document,
    EvaluationCase,
    EvaluationReport,
    Feedback,
    IndexState,
    IngestionReport,
    ParsedDocument,
    ParsedSection,
    QueryContext,
    RAGResponse,
    RawDocument,
    RetrievalResult,
    SearchHit,
    TranscriptTurn,
)
from .parsing import CsvParser, HtmlParser, JsonParser, ParserRegistry, PdfParser, TextParser
from .pipeline import RAGPipeline, RAGSession
from .processing import (
    CallableEnricher,
    DocumentDeduplicator,
    MetadataEnricher,
    StandardCleaner,
    StructureAwareChunker,
    content_fingerprint,
)
from .prompting import GroundedPromptBuilder, GroundingValidator, InvalidLLMResponse
from .protocols import (
    ChunkEnricher,
    Chunker,
    DocumentCleaner,
    DocumentParser,
    DocumentSource,
    DocumentStore,
    EmbeddingProvider,
    EventSink,
    IndexBackend,
    LLMProvider,
    QueryPlanner,
    QueryProcessor,
    Redactor,
    Reranker,
    Retriever,
)
from .query import CallableQueryProcessor, LLMQueryRewriter, NormalizeQuery, QueryProcessorChain, StaticQueryExpansion
from .retrieval import InMemoryHybridRetriever
from .retry import RetryPolicy, with_retry
from .sources import AsyncIterableSource, CallableSource, DirectorySource, IterableSource

__all__ = [
    "AsyncIterableSource",
    "AsyncTTLCache",
    "CallableEnricher",
    "CallableEmbeddingProvider",
    "CallableLLMProvider",
    "CallableQueryProcessor",
    "CallableSource",
    "Chunk",
    "ChunkEnricher",
    "Chunker",
    "Citation",
    "ContextBuilder",
    "CsvParser",
    "DirectorySource",
    "Document",
    "DocumentCleaner",
    "DocumentDeduplicator",
    "DocumentParser",
    "DocumentSource",
    "DocumentStore",
    "EmbeddingProvider",
    "EvaluationCase",
    "EvaluationReport",
    "Evaluator",
    "EventSink",
    "Feedback",
    "FileSystemDocumentStore",
    "GroundedPromptBuilder",
    "GroundingValidator",
    "HtmlParser",
    "IndexState",
    "IndexBackend",
    "IngestionConfig",
    "IngestionPipeline",
    "IngestionReport",
    "InMemoryDocumentStore",
    "InMemoryFeedbackStore",
    "InMemoryHybridRetriever",
    "InMemoryVersionedIndex",
    "InvalidLLMResponse",
    "IterableSource",
    "JsonParser",
    "LLMQueryRewriter",
    "LatestTurnQueryPlanner",
    "LLMProvider",
    "MetadataEnricher",
    "NormalizeQuery",
    "ParsedDocument",
    "ParsedSection",
    "ParserRegistry",
    "PdfParser",
    "PipelineConfig",
    "QueryContext",
    "QueryPlanner",
    "QueryProcessor",
    "QueryProcessorChain",
    "RAG",
    "RAGPipeline",
    "RAGResponse",
    "RAGSession",
    "RawDocument",
    "Redactor",
    "RegexPIIRedactor",
    "RetryPolicy",
    "RetrievalConfig",
    "RetrievalResult",
    "Retriever",
    "Reranker",
    "SearchHit",
    "StandardCleaner",
    "StaticQueryExpansion",
    "StructureAwareChunker",
    "TextParser",
    "TranscriptTurn",
    "content_fingerprint",
    "with_retry",
    "__version__",
]
