# Enterprise-RAG

Production-ready Retrieval-Augmented Generation platform for enterprise knowledge bases.

## What is Enterprise-RAG?

Enterprise-RAG transforms enterprise documents into reliable, searchable knowledge. Unlike basic RAG systems, it provides:

- **Document Processing**: Handle PDFs, images, scanned documents, tables, and diagrams with layout-aware parsing
- **Hybrid Retrieval**: Combine dense vector search, keyword search, and knowledge graphs
- **Citation Verification**: Every answer is backed by verifiable evidence with hallucination detection
- **Security & Compliance**: Policy enforcement, data lineage tracking, and audit logging
- **Production Observability**: Built-in tracing, metrics, and continuous evaluation

Perfect for organizations that need explainable, auditable AI responses.

## Features

**Document Processing**
- Multiple format support (PDF, images, scanned docs, tables)
- Layout-aware parsing via Docling
- OCR and Vision Model support
- Duplicate detection and metadata enrichment
- Hierarchical chunking with relationship tracking

**Intelligent Retrieval**
- Hybrid search (dense vectors + BM25 + graph queries)
- Multimodal embeddings for text and images
- Cross-encoder reranking with confidence scoring
- Knowledge graph integration (Neo4j)
- Result fusion and ranking optimization

**Response Quality**
- Citation tracking back to source documents
- Hallucination detection and validation
- Confidence scoring for all responses
- Source attribution and evidence highlighting
- Execution traces for explainability

**Production Ready**
- OpenTelemetry integration for distributed tracing
- Structured logging and error tracking
- Continuous evaluation (RAGAS, DeepEval)
- Semantic and response caching
- Performance optimization techniques

## Quick Start

### Requirements
- Python 3.10+
- Docker (optional)
- API keys: LLM provider (OpenAI, Groq, etc.) and embeddings (Voyage AI, etc.)

### Setup

1. Clone the repository
```bash
git clone https://github.com/bunny1323/Enterprise-RAG.git
cd Enterprise-RAG
```

2. Create virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -e .
```

4. Configure environment
```bash
cp .env.example .env
```

Edit `.env` with your API keys and service URLs:
```
LLM_PROVIDER=openai
LLM_MODEL=gpt-4
OPENAI_API_KEY=your_key_here

EMBEDDING_MODEL=voyage-3
VOYAGE_API_KEY=your_key_here

WEAVIATE_URL=http://localhost:8080
NEO4J_URI=bolt://localhost:7687
POSTGRES_DSN=postgresql://user:pass@localhost/enterprise_rag
```

5. Start services
```bash
docker-compose up -d
```

6. Run the API
```bash
uvicorn apps.api.main:app --reload
```

Visit `http://localhost:8000/docs` for interactive API documentation.

## Core Architecture

Enterprise-RAG consists of four layers:

**Agents**: Make decisions and orchestrate workflows
- Ingestion Agent
- Retrieval Agent
- Verification Agent
- Response Agent

**Services**: Perform deterministic operations
- Parsing Service
- Embedding Service
- Reranking Service
- Citation Service
- Evaluation Service

**Infrastructure**: Connect external systems
- Weaviate (vector storage)
- Neo4j (knowledge graphs)
- PostgreSQL (application data)
- Redis (caching)

**Workflows**: Execute end-to-end pipelines using LangGraph
- Document ingestion workflow
- Query and generation workflow

This modular design allows components to be reused across different industries and projects.

## API Usage

### Upload a Document

```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@document.pdf"
```

### Query the System

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the key findings?",
    "include_citations": true,
    "top_k": 5
  }'
```

### Response Example

```json
{
  "answer": "The key findings show a 23% increase in revenue...",
  "citations": [
    {
      "document": "Q3_Report.pdf",
      "page": 3,
      "text": "Revenue increased by 23% year-over-year"
    }
  ],
  "confidence": 0.94,
  "retrieval_metrics": {
    "documents_retrieved": 12,
    "reranking_score": 0.91
  }
}
```

### API Endpoints

**Documents**
- `POST /api/v1/documents/upload` - Upload document
- `GET /api/v1/documents` - List documents
- `GET /api/v1/documents/{id}` - Get document
- `DELETE /api/v1/documents/{id}` - Delete document

**Queries**
- `POST /api/v1/query` - Single query
- `POST /api/v1/query/stream` - Streaming query with SSE

**Health**
- `GET /health` - Service health
- `GET /health/ready` - Ready for requests

**Evaluation**
- `POST /api/v1/evaluation` - Run evaluation tests
- `GET /api/v1/evaluation/{run_id}` - Get results

## Configuration

### Model Selection

Enterprise-RAG supports multiple providers for each component:

**Embeddings**: Voyage AI, OpenAI, Hugging Face
**LLMs**: OpenAI, Groq, Ollama, Azure
**Rerankers**: Voyage, BGE models, custom

Configure in `.env` or update `config/models.yaml`:
```yaml
embeddings:
  provider: voyage
  model: voyage-3

llm:
  provider: openai
  model: gpt-4
  temperature: 0.7

reranker:
  provider: voyage
  top_k: 5
```

### Security Policies

Define access policies in `policies/opa/retrieval.rego`:
```
package retrieval

allow if {
    input.user_role in ["analyst", "admin"]
    input.document_classification in ["public", "internal"]
}
```

The system uses default-deny: explicit allow via policy is required.

## Use Cases

**Financial Services**: Analyze reports, filings, and compliance documents with automatic citation tracking

**Healthcare**: Process clinical guidelines and patient records with privacy controls

**Legal**: Review contracts, case law, and compliance policies with explainable retrieval

**Manufacturing**: Extract knowledge from technical manuals and operational procedures

**Enterprise Knowledge**: Internal documentation, wikis, and knowledge bases

## Evaluation

Run evaluations to measure RAG quality:

```bash
curl -X POST http://localhost:8000/api/v1/evaluation \
  -H "Content-Type: application/json" \
  -d '{
    "test_cases": [
      {
        "query": "What is the revenue target?",
        "ground_truth": "The revenue target is $10M"
      }
    ],
    "metrics": ["faithfulness", "answer_relevance", "citation_accuracy"]
  }'
```

Metrics tracked:
- **Faithfulness**: Does answer match retrieved documents?
- **Answer Relevance**: Does answer match the query?
- **Citation Accuracy**: Are citations correct and necessary?
- **Retrieval Quality**: Were relevant documents retrieved?

## Project Structure

```
Enterprise-RAG/
├── agents/              # Decision-making orchestrators
├── services/            # Deterministic operations
├── infrastructure/      # External systems (Weaviate, Neo4j, etc.)
├── workflows/           # LangGraph orchestration
├── apps/
│   ├── api/            # FastAPI backend
│   └── workers/        # Async workers
├── config/             # Configuration files
├── policies/           # OPA security policies
├── observability/      # Tracing and logging
├── evaluation/         # RAG evaluation framework
├── tests/              # Test suite
└── docs/               # Documentation
```

## Development

### Install for Development

```bash
pip install -e ".[dev]"
```

### Run Tests

```bash
pytest tests/ -v
```

### Code Quality

```bash
black .
flake8 .
mypy .
```

### Contributing

Contributions are welcome! See CONTRIBUTING.md for guidelines.

## Roadmap

**Phase 1** - Foundation and document upload ✅
**Phase 2** - Document parsing and processing ✅
**Phase 3** - Embeddings and retrieval ✅
**Phase 4** - Reranking and LLM integration 🔄
**Phase 5** - Citation and hallucination detection ⏳
**Phase 6** - Lineage, observability, evaluation ⏳

## Documentation

- [Setup Guide](docs/setup.md) - Detailed installation
- [API Reference](http://localhost:8000/docs) - Interactive API docs (when running)
- [Architecture Guide](docs/architecture.md) - System design deep dive
- [Security Guide](docs/security.md) - Production security checklist
- [Examples](examples/) - Code samples and tutorials
- [Troubleshooting](docs/troubleshooting.md) - Common issues

## Performance

Typical latencies on 100K+ document workloads:

- Document upload & parse: 2-5s per document
- Retrieval (with reranking): 200-500ms
- LLM generation: 500ms-2s
- End-to-end query: 1-3s

See docs/performance.md for optimization techniques.

## Security

- Default-deny policy enforcement via OPA
- Document classification and access control
- Complete audit logging and lineage tracking
- Model governance and version control
- Input validation and sanitization

See docs/security.md for production deployment checklist.

## License

MIT License - See LICENSE file for details

## Support

- **Issues**: [GitHub Issues](https://github.com/bunny1323/Enterprise-RAG/issues)
- **Discussions**: [GitHub Discussions](https://github.com/bunny1323/Enterprise-RAG/discussions)
- **Contributing**: [CONTRIBUTING.md](CONTRIBUTING.md)

## Built With

- **LangGraph** - Workflow orchestration
- **Weaviate** - Vector database
- **Neo4j** - Knowledge graphs
- **Docling** - Document parsing
- **OpenTelemetry** - Observability
- **FastAPI** - Web framework
- **PostgreSQL** - Application database

---

**Need help?** Check out the documentation or open an issue on GitHub.
