# Forgetting Policy Configuration Design

**Issue:** #18 — Tiered knowledge: forgetting policy configuration
**Date:** 2026-03-27
**Status:** Approved (self-contained config-only task with clear spec)

## Overview

Add a `[forgetting]` configuration section to `corvia.toml` that controls
inactivity-based and budget-based forgetting policies with a 3-level config
hierarchy: global defaults → per-memory-type overrides → per-scope overrides.

This is infrastructure-only — provides the config types that the GC worker (#19)
will consume. No runtime behavior changes.

## Config Shape (TOML)

```toml
[forgetting]
enabled = true
interval_minutes = 60

[forgetting.defaults]
max_inactive_days = 90
budget_top_n = 10000

[forgetting.by_type.episodic]
max_inactive_days = 14

[forgetting.by_type.structural]
enabled = false

[forgetting.by_type.decisional]
max_inactive_days = 365

[forgetting.by_type.procedural]
max_inactive_days = 180

# Per-scope overrides (most specific wins)
[scopes.compliance.forgetting]
max_inactive_days = 3650
budget_top_n = 0
```

## Rust Types

### Top-level: `ForgettingConfig`

```rust
pub struct ForgettingConfig {
    pub enabled: bool,                              // default: false
    pub interval_minutes: u32,                      // default: 60
    pub defaults: ForgettingPolicyConfig,            // global defaults
    pub by_type: HashMap<MemoryType, PerTypeForgettingConfig>,
}
```

### Policy values: `ForgettingPolicyConfig`

```rust
pub struct ForgettingPolicyConfig {
    pub max_inactive_days: u32,   // default: 90
    pub budget_top_n: u32,        // default: 10_000; 0 = no limit
}
```

### Per-type override: `PerTypeForgettingConfig`

```rust
pub struct PerTypeForgettingConfig {
    pub enabled: Option<bool>,              // None = inherit from global
    pub max_inactive_days: Option<u32>,     // None = inherit
    pub budget_top_n: Option<u32>,          // None = inherit
}
```

### Per-scope override: `ScopeForgettingOverride`

Same shape as `PerTypeForgettingConfig` — all fields optional, merged on top.

### Resolved output: `ResolvedPolicy`

```rust
pub struct ResolvedPolicy {
    pub enabled: bool,
    pub max_inactive_days: u32,
    pub budget_top_n: u32,
}
```

## Resolution Algorithm

`resolve_policy(scope_id, memory_type) -> ResolvedPolicy`:

1. Start with global `defaults`
2. If `forgetting.enabled == false`, return `ResolvedPolicy { enabled: false, .. }`
3. Layer per-type override: `by_type[memory_type]` fields replace non-None values
4. If per-type `enabled == Some(false)`, return disabled
5. Layer per-scope override: `scope[scope_id].forgetting` fields replace non-None values
6. Return merged `ResolvedPolicy`

Most-specific wins at each field level. Missing fields inherit from the layer below.

## Validation Rules

- `max_inactive_days > 0` (zero would mean immediate forgetting)
- `budget_top_n >= 0` (0 means no budget limit)
- `interval_minutes > 0`
- Per-type keys must be valid `MemoryType` variants (enforced by serde)

## Integration Points

- `ScopeConfig` gets an optional `forgetting: Option<ScopeForgettingOverride>` field
- `CorviaConfig` gets an optional `forgetting: Option<ForgettingConfig>` field
- Backward compatible: missing `[forgetting]` section = no forgetting (disabled)

## Files to Modify

1. `crates/corvia-common/src/config.rs` — All new types + `resolve_policy()` + tests
2. No changes to `types.rs` — `MemoryType` already supports `Hash` + serde
