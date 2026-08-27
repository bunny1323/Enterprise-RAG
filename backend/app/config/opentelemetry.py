"""
OpenTelemetry instrumentation for Enterprise-RAG.

Exports:
  - Traces via OTLP (production) or Console (local dev)
  - Prometheus-compatible metrics via prometheus_client

Usage:
  Call `setup_telemetry()` once at application startup (in lifespan).
  Use `get_tracer()` and `get_meter()` to instrument code.
"""
from __future__ import annotations

import os
from typing import Optional

from app.config.logging import get_logger

logger = get_logger(__name__)

# ── Lazy imports to avoid hard dependency when OTEL packages are not installed ──

_tracer = None
_meter = None
_initialized = False


def setup_telemetry(
    service_name: str = "enterprise-rag",
    otlp_endpoint: Optional[str] = None,
    enable_console: bool = False,
) -> None:
    """
    Initialize OpenTelemetry trace and metric providers.

    Args:
        service_name:    Service name reported in traces.
        otlp_endpoint:   OTLP gRPC/HTTP endpoint (e.g., 'http://localhost:4317').
                         If None, reads from OTEL_EXPORTER_OTLP_ENDPOINT env var.
        enable_console:  If True, also export traces to console (useful in dev).
    """
    global _tracer, _meter, _initialized

    if _initialized:
        return

    try:
        from opentelemetry import trace, metrics
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource

        resource = Resource.create({"service.name": service_name})

        # ── Trace Provider ──────────────────────────────────────────────────────
        tracer_provider = TracerProvider(resource=resource)

        endpoint = otlp_endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "")
        if endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
                otlp_exporter = OTLPSpanExporter(endpoint=endpoint)
                tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
                logger.info("otel.otlp_trace_exporter_enabled", endpoint=endpoint)
            except ImportError:
                logger.warning("otel.otlp_exporter_not_installed_trying_http")
                try:
                    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as HTTPSpanExporter
                    otlp_exporter = HTTPSpanExporter(endpoint=endpoint)
                    tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
                    logger.info("otel.otlp_http_trace_exporter_enabled", endpoint=endpoint)
                except ImportError:
                    logger.warning("otel.otlp_exporter_unavailable_no_traces")

        if enable_console or not endpoint:
            tracer_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
            logger.info("otel.console_trace_exporter_enabled")

        trace.set_tracer_provider(tracer_provider)

        # ── Metric Provider ─────────────────────────────────────────────────────
        metric_readers = []
        if endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
                metric_readers.append(
                    PeriodicExportingMetricReader(OTLPMetricExporter(endpoint=endpoint), export_interval_millis=60000)
                )
            except ImportError:
                pass

        if not metric_readers or enable_console:
            metric_readers.append(PeriodicExportingMetricReader(ConsoleMetricExporter(), export_interval_millis=60000))

        meter_provider = MeterProvider(resource=resource, metric_readers=metric_readers)
        metrics.set_meter_provider(meter_provider)

        _tracer = trace.get_tracer(service_name)
        _meter = metrics.get_meter(service_name)

        _create_metrics(_meter)

        _initialized = True
        logger.info("otel.setup_complete", service=service_name)

    except ImportError as e:
        logger.warning("otel.not_installed_skipping", detail=str(e))
        _initialized = True  # Mark done to avoid repeated attempts


def get_tracer():
    """Return the application tracer (or a no-op tracer if OTEL is unavailable)."""
    try:
        from opentelemetry import trace
        return trace.get_tracer("enterprise-rag")
    except ImportError:
        return _NoOpTracer()


def get_meter():
    """Return the application meter (or None if OTEL is unavailable)."""
    try:
        from opentelemetry import metrics
        return metrics.get_meter("enterprise-rag")
    except ImportError:
        return None


# ── Application Metrics ─────────────────────────────────────────────────────────

_metrics: dict = {}


def _create_metrics(meter) -> None:
    """Create all application-level Prometheus-compatible metrics."""
    global _metrics
    _metrics = {
        "request_count": meter.create_counter(
            "rag_request_total", description="Total RAG query requests"
        ),
        "error_count": meter.create_counter(
            "rag_error_total", description="Total RAG errors"
        ),
        "cache_hit": meter.create_counter(
            "rag_cache_hits_total", description="Cache hits"
        ),
        "cache_miss": meter.create_counter(
            "rag_cache_misses_total", description="Cache misses"
        ),
        "retrieval_latency": meter.create_histogram(
            "rag_retrieval_latency_ms", description="Retrieval latency in ms", unit="ms"
        ),
        "generation_latency": meter.create_histogram(
            "rag_generation_latency_ms", description="LLM generation latency in ms", unit="ms"
        ),
        "rerank_latency": meter.create_histogram(
            "rag_rerank_latency_ms", description="Reranking latency in ms", unit="ms"
        ),
        "parse_latency": meter.create_histogram(
            "rag_parse_latency_ms", description="Document parse latency in ms", unit="ms"
        ),
        "embed_latency": meter.create_histogram(
            "rag_embed_latency_ms", description="Embedding latency in ms", unit="ms"
        ),
        "token_usage": meter.create_counter(
            "rag_token_usage_total", description="Total LLM tokens used"
        ),
        "retry_count": meter.create_counter(
            "rag_retries_total", description="Total retries across all operations"
        ),
        "ingestion_count": meter.create_counter(
            "rag_ingestion_total", description="Documents ingested"
        ),
    }


def record_request(tenant_id: str = "unknown") -> None:
    c = _metrics.get("request_count")
    if c:
        c.add(1, {"tenant": tenant_id})


def record_error(error_type: str = "unknown", tenant_id: str = "unknown") -> None:
    c = _metrics.get("error_count")
    if c:
        c.add(1, {"error_type": error_type, "tenant": tenant_id})


def record_cache_hit(tenant_id: str = "unknown") -> None:
    c = _metrics.get("cache_hit")
    if c:
        c.add(1, {"tenant": tenant_id})


def record_cache_miss(tenant_id: str = "unknown") -> None:
    c = _metrics.get("cache_miss")
    if c:
        c.add(1, {"tenant": tenant_id})


def record_retrieval_latency(latency_ms: float, strategy: str = "hybrid") -> None:
    h = _metrics.get("retrieval_latency")
    if h:
        h.record(latency_ms, {"strategy": strategy})


def record_generation_latency(latency_ms: float, provider: str = "unknown") -> None:
    h = _metrics.get("generation_latency")
    if h:
        h.record(latency_ms, {"provider": provider})


def record_token_usage(tokens: int, tenant_id: str = "unknown") -> None:
    c = _metrics.get("token_usage")
    if c:
        c.add(tokens, {"tenant": tenant_id})


def record_ingestion(tenant_id: str = "unknown") -> None:
    c = _metrics.get("ingestion_count")
    if c:
        c.add(1, {"tenant": tenant_id})


# ── No-op fallback ──────────────────────────────────────────────────────────────

class _NoOpSpan:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def set_attribute(self, *args):
        pass
    def record_exception(self, *args):
        pass
    def set_status(self, *args):
        pass


class _NoOpTracer:
    def start_as_current_span(self, name, **kwargs):
        return _NoOpSpan()
    def start_span(self, name, **kwargs):
        return _NoOpSpan()
