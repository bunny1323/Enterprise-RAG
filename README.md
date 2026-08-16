Enterprise-RAG

A modular, production-oriented Enterprise Retrieval-Augmented Generation (RAG) platform designed to process enterprise documents, retrieve reliable evidence, and generate citation-backed, explainable responses.

The architecture separates Agents, Services, Pipelines, and Infrastructure so individual components can be reused across different industries and projects.

Overview

Enterprise-RAG is designed to support complex enterprise knowledge bases containing:

Text documents
Tables
Images
Technical diagrams
Scanned documents
Structured and unstructured data

The system combines layout-aware document processing, hybrid retrieval, reranking, knowledge graphs, multimodal understanding, explainability, security, and evaluation.

Core Architecture
Documents
   ↓
Validation & Deduplication
   ↓
Docling / OCR / Vision
   ↓
Structure-Aware Chunking
   ↓
Metadata & Relationships
   ↓
Embeddings
   ↓
Weaviate + Neo4j + PostgreSQL
   ↓
Hybrid Retrieval
   ↓
Reranking
   ↓
Confidence Evaluation
   ↓
LLM Generation
   ↓
Citation & Hallucination Verification
   ↓
Policy & Explainability
   ↓
Final Response
Key Features
Layout-aware document ingestion
PDF validation and duplicate detection
SHA-256 document hashing
Docling-based document parsing
OCR for scanned documents
Vision-Language Model support for diagrams
Structure-aware hierarchical chunking
Rich document and chunk metadata
Multimodal embeddings
Hybrid vector + keyword retrieval
Knowledge Graph / Graph RAG
ANN-based vector search
Cross-encoder reranking
Confidence scoring
Citation verification
Hallucination detection
Source attribution
Policy enforcement with OPA
Data and model lineage
Observability and tracing
Continuous RAG evaluation
Caching and memory
LangGraph-based orchestration
Architecture
1. Document Ingestion
Upload
  ↓
Validation
  ↓
Document Inventory
  ↓
Hash & Version Detection
  ↓
Duplicate Check
  ↓
Parsing
Parsing

The parsing layer can use:

Docling — primary layout-aware parser
PyMuPDF — lightweight PDF extraction/fallback
Tesseract / EasyOCR — OCR when required
Vision-Language Models — diagrams, schematics and complex visual content
2. Document Understanding
Parsed Document
      ↓
Text ────────┐
Tables ──────┤
Images ──────┤
Diagrams ────┘
      ↓
Structure-Aware Chunking
      ↓
Metadata Enrichment
      ↓
Relationship Extraction

Chunks preserve contextual information such as:

Document ID
Page number
Section
Subsection
Figure ID
Table ID
Caption
Bounding box
Parent/child relationships
Content type
Security classification
3. Knowledge & Indexing Layer

Enterprise-RAG uses specialized storage systems for different responsibilities.

Technology	Purpose
Weaviate	Vector storage and hybrid retrieval
Neo4j	Knowledge graph and relationships
PostgreSQL / Supabase	Application metadata, audit and lineage
Redis	Cache and short-term state
Vector Indexing

The system can maintain representations for:

Text
Tables
Images
Diagram descriptions

This enables multimodal retrieval instead of relying only on plain text.

4. Retrieval Pipeline
User Query
    ↓
Query Understanding
    ↓
Security / Policy Check
    ↓
┌──────────────┬──────────────┬──────────────┐
│ Dense Search │ BM25 Search  │ Graph Search │
└──────────────┴──────────────┴──────────────┘
              ↓
          Result Fusion
              ↓
        ANN Candidate Search
              ↓
           Reranking
              ↓
       Confidence Scoring
Retrieval Methods
Dense vector retrieval
BM25 / sparse retrieval
Graph retrieval
Multimodal retrieval
Hybrid retrieval
Reciprocal Rank Fusion
Approximate Nearest Neighbor search
Cross-encoder reranking
5. Generation & Verification
Retrieved Evidence
       ↓
Confidence Check
       ↓
LLM Generation
       ↓
Citation Mapping
       ↓
Hallucination Detection
       ↓
Policy Validation
       ↓
Final Response

The response should be generated from retrieved evidence rather than unsupported model knowledge.

6. Explainability & Trust

Enterprise-RAG provides multiple mechanisms for understanding and validating responses.

Source Attribution

Maps generated claims back to their supporting:

Document
   ↓
Page
   ↓
Chunk
   ↓
Evidence
Explainability

Includes:

Source attribution
Evidence highlighting
Confidence scores
Citation verification
Retrieval information
Model information
Execution trace
7. Security & Governance
Open Policy Agent

OPA can enforce policies such as:

User
 ↓
Authorization
 ↓
Document Security Metadata
 ↓
Policy Evaluation
 ↓
Allowed Context

The system follows a default-deny approach when required security information is missing.

Lineage
Data Lineage

Tracks:

Source Document
      ↓
Parsed Content
      ↓
Chunk
      ↓
Embedding
      ↓
Retrieved Evidence
      ↓
Generated Response
Model Lineage

Tracks information such as:

Embedding model
Reranker
LLM
Vision model
Model version
Configuration
Execution metadata
Agent Architecture

Agents are used only where decision-making or orchestration is required.

Agents
Supervisor Agent
Validation Agent
Duplicate Detection Agent
Parsing Agent
Chunking Agent
Metadata Agent
Embedding Agent
Indexing Agent
Retrieval Agent
Reranking Agent
Confidence Agent
LLM/Answer Agent
Citation Agent
Hallucination Detection Agent
Policy Agent
Response Agent
Observability Agent
Evaluation Agent
Lineage Agent
Services

Services perform deterministic capabilities.

Examples:

Parser Service
OCR Service
Vision Service
Embedding Service
Chunking Service
Dense Search Service
BM25 Service
Graph Search Service
Reranking Service
Cache Service
LLM Provider Service
Citation Service
Evaluation Service
Principle

Agents make decisions. Services perform capabilities. Pipelines execute predictable steps. Infrastructure connects external systems.

This prevents unnecessary agent proliferation and keeps the architecture modular.

Orchestration

LangGraph is used for workflow orchestration where state, routing, conditional execution and retries are required.

Example:

Supervisor
    ↓
Retrieval
    ↓
Reranking
    ↓
Confidence
    │
    ├── High → Generation
    │
    └── Low → Query Expansion → Retrieval
                         ↓
                    Generation
                         ↓
                  Hallucination Check
                         │
                 ┌───────┴───────┐
                 ↓               ↓
              Valid          Invalid
                 ↓               ↓
              Response       Regenerate
Models

The architecture is model-provider independent.

Embeddings

Primary embedding layer:

Voyage AI embedding models
Multimodal embedding support where required

Fallback models can be configured independently.

Reranking

Possible implementation:

Voyage reranker
BGE reranker models
LLM

The LLM layer can support multiple providers, for example:

Groq
Ollama
Other compatible LLM providers
Vision

Vision processing is separated from the normal text LLM when visual understanding is required.

Possible providers include:

Qwen-VL family
Other compatible Vision-Language Models
Performance

The architecture supports inference and retrieval optimization techniques such as:

Semantic caching
Response caching
Embedding caching
KV caching where supported by the inference runtime
Quantization where supported
Efficient ANN indexing
Batch processing
Incremental ingestion
Connection pooling
Memory-aware execution

Optimization techniques should be enabled based on the selected model/runtime rather than being forced into every deployment.

Evaluation & Observability

The system is designed for continuous monitoring and evaluation.

Observability
OpenTelemetry
Distributed tracing
Structured logging
Latency metrics
Token usage
Retrieval metrics
Agent execution metrics
Error tracking
Evaluation

Potential evaluation frameworks:

RAGAS
DeepEval

Metrics can include:

Faithfulness
Context relevance
Answer relevance
Retrieval quality
Citation accuracy
Hallucination rate
Project Structure
Enterprise-RAG/
│
├── apps/
│   ├── api/
│   └── workers/
│
├── agents/
│   ├── supervisor/
│   ├── ingestion/
│   ├── retrieval/
│   ├── verification/
│   └── response/
│
├── services/
│   ├── ingestion/
│   ├── parsing/
│   ├── ocr/
│   ├── vision/
│   ├── chunking/
│   ├── metadata/
│   ├── embeddings/
│   ├── retrieval/
│   ├── reranking/
│   ├── citation/
│   ├── hallucination/
│   └── evaluation/
│
├── infrastructure/
│   ├── weaviate/
│   ├── neo4j/
│   ├── postgres/
│   ├── redis/
│   └── llm/
│
├── workflows/
│   ├── ingestion/
│   └── query/
│
├── models/
│
├── schemas/
│
├── policies/
│   └── opa/
│
├── lineage/
│
├── observability/
│
├── evaluation/
│
├── tests/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── frontend/
│
├── .env.example
├── pyproject.toml
├── README.md
└── docker-compose.yml
API Overview
Document APIs
POST   /api/v1/documents/upload
GET    /api/v1/documents
GET    /api/v1/documents/{document_id}
DELETE /api/v1/documents/{document_id}
Query APIs
POST /api/v1/query
POST /api/v1/query/stream
Health
GET /health
GET /health/ready
Evaluation
POST /api/v1/evaluation
GET  /api/v1/evaluation/{run_id}
Setup

Clone the repository:

git clone https://github.com/bunny1323/Enterprise-RAG.git
cd Enterprise-RAG

Create a virtual environment:

python -m venv .venv

Activate it on Windows:

.venv\Scripts\activate

Install dependencies:

pip install -e .

Create environment configuration:

copy .env.example .env

Configure the required services and API keys in .env.

Development

Run the backend according to the project's configured entry point.

Example:

uvicorn apps.api.main:app --reload

API documentation:

/docs
Development Principles

The project follows these principles:

Modular architecture
Reusable agents and services
Provider-independent model layer
Evidence-first generation
Citation-backed responses
Security-aware retrieval
Observable workflows
Continuous evaluation
Incremental document processing
Avoid unnecessary agents
Current Direction

Enterprise-RAG is being developed as a generic enterprise RAG platform, rather than being tied to a single domain.

The same architecture can support:

Healthcare
Finance
Manufacturing
Legal
Education
Research
Enterprise Knowledge Bases
Technical Documentation

Domain-specific behavior should primarily come from:

Configuration
Metadata taxonomies
Retrieval strategies
Prompts
Policies
Knowledge graphs
Model selection

rather than hardcoded business logic.

Roadmap
Phase 1
Project foundation
Configuration
Document upload
Validation
Document inventory
Hashing and duplicate detection
Phase 2
Docling integration
OCR
Vision processing
Structure-aware chunking
Metadata enrichment
Phase 3
Voyage embeddings
Weaviate indexing
Neo4j knowledge graph
Hybrid retrieval
Phase 4
Reranking
Confidence scoring
LangGraph orchestration
LLM integration
Phase 5
Citation verification
Hallucination detection
Explainability
OPA policies
Phase 6
Lineage
Observability
Continuous evaluation
Caching
Performance optimization
Project Goal

Build a reusable, explainable, secure, and production-oriented Enterprise RAG platform capable of transforming heterogeneous enterprise documents into reliable, searchable knowledge and generating grounded responses with verifiable evidence.

License

Add the project's selected license here.

Contributors

Enterprise-RAG is developed collaboratively with a modular architecture intended to support future contributors and domain-specific implementations.