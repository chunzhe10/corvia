+++
id = "fixture-learning-01"
created_at = "2026-04-02T10:00:00Z"
kind = "learning"
tags = ["redb", "gotcha"]
+++

Redb uses file-level locking. If two processes try to open the same database for writing, the second process blocks until the first commits. This means corvia write and corvia mcp should not run concurrently.
