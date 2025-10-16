from __future__ import annotations

import os
from celery import Celery


def make_celery() -> Celery:
    broker_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    backend_url = os.environ.get("CELERY_RESULT_BACKEND", broker_url)
    app = Celery(
        "datagen",
        broker=broker_url,
        backend=backend_url,
        include=["datagen.tasks"],
    )
    # Reasonable defaults
    app.conf.update(
        task_acks_late=True,
        worker_prefetch_multiplier=int(os.environ.get("CELERY_PREFETCH", "1")),
        task_time_limit=int(os.environ.get("CELERY_TASK_TIME_LIMIT", "1200")),
        task_soft_time_limit=int(os.environ.get("CELERY_TASK_SOFT_TIME_LIMIT", "900")),
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
    )
    return app


celery_app = make_celery()


