+++
id = "fixture-decision-01"
created_at = "2026-04-01T10:00:00Z"
kind = "decision"
tags = ["storage", "architecture"]
+++

We chose Redb over SQLite for the index store. Redb is pure Rust, has ACID transactions, and requires no system dependencies. SQLite would have added a C compilation requirement.
