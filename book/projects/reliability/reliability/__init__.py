# path: book/projects/reliability/reliability/__init__.py
"""reliability: system-level reliability primitives for AI services (Chapter 29).

Single model calls are protected by aie_core's ModelGateway (Chapter 3). This package
protects everything around them: deadlines across stages, circuit breakers per dependency,
bulkheads per workload, admission control and load shedding, retry budgets, durable job
queues and workers, degraded modes, malformed-output recovery, partial-failure chains,
SLO arithmetic, and fault injection for chaos tests.
"""
from .admission import (
    Action,
    AdmissionConfig,
    AdmissionController,
    AdmissionDecision,
    AdmissionRequest,
    Priority,
    TenantQuota,
    TokenBucket,
)
from .bulkhead import AsyncBulkhead, Bulkhead, Bulkheads, BulkheadStats
from .chain import ChainResult, ChainStatus, Step, StepRecord, StepStatus, run_chain
from .chaos import ChaosFunction, ChaosLLM, Fault, FaultPlan, FaultWindow, malformed, outage, rate_limited, slow, timeouts
from .circuit import CircuitBreaker, CircuitBreakerClient, CircuitBreakerRegistry, CircuitState
from .clock import ManualClock
from .deadline import Deadline, current_deadline
from .degrade import DEFAULT_PLANS, DegradedPlan, DegradeLevel, DegradePolicy
from .errors import (
    AdmissionRejected,
    BulkheadFullError,
    CircuitOpenError,
    DeadlineExceeded,
    Overloaded,
    is_retryable,
)
from .malformed import Outcome, Recovered, close_truncated_json, complete_with_recovery, salvage_partial
from .queue import InMemoryJobQueue, Job, JobQueue, JobState, Lease, RedisJobQueue
from .retry import RetryBudget, acall_with_retry, ahedged, call_with_retry
from .slo import RequestOutcome, SLIReport, SLOTargets, burn_rate, evaluate, should_page
from .worker import (
    InMemoryIdempotencyStore,
    JobContext,
    LeaseLost,
    PermanentJobError,
    ShutdownRequested,
    Worker,
    WorkOutcome,
    WorkResult,
    once,
)

__version__ = "0.1.0"
__all__ = [name for name in dir() if not name.startswith("_")]
