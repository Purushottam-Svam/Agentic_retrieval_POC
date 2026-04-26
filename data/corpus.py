"""
Enterprise knowledge corpus for TechNova Inc.
Designed so that:
  - Simple queries work well with both semantic and agentic retrieval
  - Complex multi-hop / aggregation queries expose the gap
"""

DOCUMENTS = [
    # ── Incident Reports ──────────────────────────────────────────────────────
    {
        "id": "INC-001",
        "title": "Q4 2024 Latency Incident Report",
        "category": "incident",
        "content": (
            "On November 14, 2024, TechNova experienced a critical latency spike in the "
            "checkout service between 14:00-16:30 UTC. P99 latency rose from 120 ms to "
            "4200 ms. Root cause: a mis-configured connection pool in the Payment Gateway "
            "service (version 3.7.2) released earlier that day. The Payment Gateway team, "
            "led by Sarah Chen, rolled back to v3.7.1 at 16:45 UTC, restoring normal "
            "latency within 10 minutes. Estimated revenue impact: $48,000. Post-incident "
            "action items: add connection-pool exhaustion alerts, mandatory load testing "
            "before payment service deployments."
        ),
        "metadata": {"date": "2024-11-14", "severity": "P1", "team": "Platform"},
    },
    {
        "id": "INC-002",
        "title": "Auth Service Outage - March 2025",
        "category": "incident",
        "content": (
            "On March 3, 2025, the Authentication service went completely down for 22 "
            "minutes (09:12-09:34 UTC) due to a Redis cache flush triggered accidentally "
            "by a runbook mistake during routine maintenance. All user login attempts "
            "failed during this window. The Auth team (lead: David Park) implemented a "
            "circuit-breaker pattern post-incident to prevent cascading failures. "
            "Recovery time for the Auth service: 22 minutes. Affected users: ~85,000."
        ),
        "metadata": {"date": "2025-03-03", "severity": "P1", "team": "Auth"},
    },
    {
        "id": "INC-003",
        "title": "Data Pipeline Slowdown - January 2025",
        "category": "incident",
        "content": (
            "Between January 7-9, 2025, the Analytics data pipeline processed jobs at "
            "30% normal throughput due to a Spark configuration regression. No customer-"
            "facing impact, but internal dashboards were delayed by up to 6 hours. "
            "The Data Engineering team (lead: Priya Mehta) resolved the issue by "
            "reverting Spark executor memory settings. Recovery time: 48 hours."
        ),
        "metadata": {"date": "2025-01-07", "severity": "P2", "team": "Data Engineering"},
    },

    # ── Architecture & Service Docs ───────────────────────────────────────────
    {
        "id": "ARCH-001",
        "title": "Payment Gateway Service - Architecture Overview",
        "category": "architecture",
        "content": (
            "The Payment Gateway service handles all transaction processing for TechNova. "
            "It integrates with Stripe (primary), Braintree (fallback), and internal "
            "fraud detection. The service is owned by the Payments team, led by "
            "Sarah Chen (sarah.chen@technova.io). SLA: 99.95% uptime. "
            "Estimated recovery time objective (RTO) in case of full failure: 15 minutes. "
            "Recovery point objective (RPO): 30 seconds (via Kafka event replay). "
            "Tech stack: Go, PostgreSQL, Redis, Kafka. Current version: 3.7.3."
        ),
        "metadata": {"team": "Payments", "lead": "Sarah Chen", "criticality": "critical"},
    },
    {
        "id": "ARCH-002",
        "title": "Authentication Service - Architecture Overview",
        "category": "architecture",
        "content": (
            "The Authentication service manages user sessions, OAuth2 flows, and API "
            "key validation. It is owned by the Auth team, led by David Park "
            "(david.park@technova.io). SLA: 99.99% uptime. "
            "Estimated recovery time objective (RTO): 10 minutes. "
            "Recovery point objective (RPO): 0 seconds (stateless JWT, Redis session "
            "cache only). Tech stack: Python (FastAPI), Redis, PostgreSQL. "
            "Circuit-breaker pattern implemented post March 2025 incident."
        ),
        "metadata": {"team": "Auth", "lead": "David Park", "criticality": "critical"},
    },
    {
        "id": "ARCH-003",
        "title": "Checkout Service - Architecture Overview",
        "category": "architecture",
        "content": (
            "The Checkout service orchestrates the end-to-end purchase flow: cart "
            "validation -> inventory check -> payment -> order creation. It calls the "
            "Payment Gateway and Auth services synchronously. Owned by the Commerce "
            "team, led by Alex Rivera (alex.rivera@technova.io). "
            "SLA: 99.9% uptime. RTO: 20 minutes. Tech stack: Node.js, MongoDB, Redis."
        ),
        "metadata": {"team": "Commerce", "lead": "Alex Rivera", "criticality": "critical"},
    },
    {
        "id": "ARCH-004",
        "title": "Analytics Data Pipeline - Architecture Overview",
        "category": "architecture",
        "content": (
            "The Analytics pipeline ingests ~2 TB/day of event data using Kafka -> Spark -> "
            "Snowflake. Owned by the Data Engineering team, led by Priya Mehta "
            "(priya.mehta@technova.io). SLA: best-effort (internal tooling). "
            "RTO: 4 hours. RPO: 1 hour. Not customer-facing."
        ),
        "metadata": {"team": "Data Engineering", "lead": "Priya Mehta", "criticality": "medium"},
    },

    # ── Policies ──────────────────────────────────────────────────────────────
    {
        "id": "POL-001",
        "title": "Production Deployment Policy",
        "category": "policy",
        "content": (
            "All production deployments at TechNova must follow this process: "
            "(1) Create a deployment ticket in Jira tagged 'prod-deploy'. "
            "(2) Get approval from the service's team lead AND one member of the "
            "Platform Reliability team. For critical services (Payment, Auth, Checkout) "
            "a second approval from the VP of Engineering (James Liu) is required. "
            "(3) Deployments are blocked on Fridays after 15:00 UTC and all day "
            "Saturday-Sunday unless a P1 incident requires an emergency fix. "
            "(4) Deployment window: Monday-Thursday 09:00-17:00 UTC, Friday 09:00-15:00 UTC. "
            "(5) A rollback plan must be documented before deployment starts."
        ),
        "metadata": {"owner": "Platform Reliability", "last_updated": "2025-01-15"},
    },
    {
        "id": "POL-002",
        "title": "Database Migration Policy",
        "category": "policy",
        "content": (
            "Database migrations that alter existing tables require: "
            "(1) A migration script reviewed by the DBA team (lead: Omar Hassan, omar.hassan@technova.io). "
            "(2) A dry-run executed on the staging environment with row-count verification. "
            "(3) Approval from the service owner and the DBA lead. "
            "(4) Migrations on tables >10M rows must be run with pt-online-schema-change "
            "or a zero-downtime equivalent. "
            "(5) Emergency rollback scripts must be tested before the migration window."
        ),
        "metadata": {"owner": "DBA Team", "last_updated": "2025-02-01"},
    },
    {
        "id": "POL-003",
        "title": "On-Call and Escalation Policy",
        "category": "policy",
        "content": (
            "TechNova on-call rotation follows PagerDuty schedules per team. "
            "P1 incidents escalate to the team lead within 5 minutes, then to the "
            "Engineering Director within 15 minutes if unresolved. "
            "P2 incidents: team lead notified within 30 minutes. "
            "The Platform Reliability team (PRE) is always secondary on-call for all "
            "critical services. PRE lead: Marcus Webb (marcus.webb@technova.io). "
            "All P1 incidents must have an Incident Commander designated within 10 minutes."
        ),
        "metadata": {"owner": "Platform Reliability", "last_updated": "2025-03-01"},
    },

    # ── Team Directory ────────────────────────────────────────────────────────
    {
        "id": "TEAM-001",
        "title": "Engineering Team Directory",
        "category": "team",
        "content": (
            "TechNova Engineering Team Directory (April 2025): "
            "VP Engineering: James Liu (james.liu@technova.io). "
            "Payments Team Lead: Sarah Chen (sarah.chen@technova.io) - owns Payment Gateway. "
            "Auth Team Lead: David Park (david.park@technova.io) - owns Authentication service. "
            "Commerce Team Lead: Alex Rivera (alex.rivera@technova.io) - owns Checkout service. "
            "Data Engineering Lead: Priya Mehta (priya.mehta@technova.io) - owns Analytics pipeline. "
            "Platform Reliability Lead: Marcus Webb (marcus.webb@technova.io). "
            "DBA Lead: Omar Hassan (omar.hassan@technova.io)."
        ),
        "metadata": {"category": "directory", "last_updated": "2025-04-01"},
    },

    # ── SLA / Runbooks ────────────────────────────────────────────────────────
    {
        "id": "RUN-001",
        "title": "Payment Gateway Runbook - Connection Pool Exhaustion",
        "category": "runbook",
        "content": (
            "Symptom: Payment Gateway returns HTTP 503 or high latency (>500ms P99). "
            "Immediate steps: (1) Check pgbouncer pool stats: pgbouncer stats show pool_size. "
            "(2) If pool_size at max, increase max_client_conn temporarily via config reload. "
            "(3) Check for long-running transactions blocking connections: "
            "SELECT * FROM pg_stat_activity WHERE wait_event_type = 'Lock'. "
            "(4) If version was recently deployed, consider rollback. Contact Sarah Chen "
            "if escalation needed. Rollback estimated time: 5 minutes. "
            "Root cause investigation should happen post-recovery."
        ),
        "metadata": {"service": "Payment Gateway", "team": "Payments"},
    },
    {
        "id": "RUN-002",
        "title": "Auth Service Runbook - Redis Cache Failure",
        "category": "runbook",
        "content": (
            "Symptom: All logins failing, Auth service returning 500 errors. "
            "Immediate steps: (1) Check Redis cluster health: redis-cli cluster info. "
            "(2) If Redis is down, Auth service should fall back to database-backed sessions "
            "(circuit breaker should handle this post-March 2025 fix). "
            "(3) If circuit breaker not triggering, manually set AUTH_FALLBACK_MODE=true "
            "in the service config and restart. "
            "(4) Contact David Park for escalation. Recovery time: ~10 minutes with circuit breaker."
        ),
        "metadata": {"service": "Authentication", "team": "Auth"},
    },

    # ── Q&A / FAQ ─────────────────────────────────────────────────────────────
    {
        "id": "FAQ-001",
        "title": "FAQ: How to request a new AWS resource",
        "category": "faq",
        "content": (
            "To request a new AWS resource: (1) Open a ticket in the #infra-requests Slack "
            "channel using the /infra-request slash command. (2) Specify the resource type, "
            "estimated cost, team, and business justification. (3) Platform Reliability team "
            "reviews within 2 business days. For urgent requests, ping Marcus Webb directly. "
            "Cost approval thresholds: <$500/month auto-approved; $500-$5000 needs team lead "
            "sign-off; >$5000 needs VP Engineering approval."
        ),
        "metadata": {"category": "infra", "owner": "Platform Reliability"},
    },
    {
        "id": "FAQ-002",
        "title": "FAQ: Deployment freeze dates 2025",
        "category": "faq",
        "content": (
            "Planned deployment freeze windows for 2025: "
            "Black Friday/Cyber Monday: Nov 24 - Dec 2, 2025 (no prod deployments). "
            "Year-end freeze: Dec 20, 2025 - Jan 5, 2026. "
            "Q2 feature freeze: June 13-20, 2025 (no new features, hotfixes only). "
            "All freeze dates apply to production environments. Staging deployments unaffected."
        ),
        "metadata": {"category": "deployment", "owner": "Platform Reliability"},
    },
    {
        "id": "FAQ-003",
        "title": "FAQ: What happens during a P1 incident?",
        "category": "faq",
        "content": (
            "During a P1 incident: (1) PagerDuty fires for the on-call engineer within 2 min. "
            "(2) On-call creates an incident Slack channel #inc-YYYYMMDD-<service>. "
            "(3) Incident Commander is designated from Platform Reliability (PRE). "
            "(4) Status page (status.technova.io) is updated within 10 minutes. "
            "(5) Customer-facing communication goes out within 15 minutes via email. "
            "(6) Engineering Director is looped in if unresolved after 15 minutes. "
            "Post-incident: blameless postmortem within 5 business days."
        ),
        "metadata": {"category": "incident-response"},
    },
]
