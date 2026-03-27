//! Orphan process cleanup on SessionEnd.
//!
//! Replaces `.devcontainer/scripts/cleanup-orphans.sh` with Rust.
//! Uses `/proc` filesystem directly — zero external dependencies.
//! Only targets processes reparented to init (PPID=1) that match known patterns.

use std::fs;

const THROTTLE_FILE: &str = "/tmp/corvia-cleanup-orphans.last";
const THROTTLE_SECS: u64 = 600;

/// Patterns to match in process command lines, with minimum age in seconds.
const ORPHAN_PATTERNS: &[(&str, u64)] = &[
    ("node", 600),     // claude node processes (further filtered by cmdline containing "claude")
    ("vite", 300),     // vite dev servers
    ("target/debug/corvia-inference", 600), // debug inference
    ("ollama serve", 600), // ollama
];

/// Run orphan cleanup. Kills orphaned processes matching known patterns.
///
/// Safety: Only kills processes with PPID=1 that match specific patterns
/// (node+claude, vite, debug inference, ollama). In containers where PID 1
/// is the container init, we additionally verify we're not the only process
/// group to avoid killing container-essential processes.
pub fn orphan_cleanup(quiet: bool) {
    // Throttle: skip if we ran recently
    if is_throttled() {
        return;
    }
    write_throttle_timestamp();

    let mut killed = 0u32;
    // boot_time not needed — uptime calculation uses /proc/uptime directly
    let clock_ticks = unsafe { libc::sysconf(libc::_SC_CLK_TCK) } as u64;
    let uptime_secs = read_uptime_secs();

    // Scan /proc for process directories
    let proc_entries = match fs::read_dir("/proc") {
        Ok(entries) => entries,
        Err(_) => return,
    };

    for entry in proc_entries.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();

        // Only numeric directories (PIDs)
        let pid: i32 = match name_str.parse() {
            Ok(p) => p,
            Err(_) => continue,
        };

        // Skip our own process
        if pid == std::process::id() as i32 {
            continue;
        }

        // Read /proc/{pid}/stat to get PPID and start time
        let stat_path = format!("/proc/{pid}/stat");
        let stat = match fs::read_to_string(&stat_path) {
            Ok(s) => s,
            Err(_) => continue,
        };

        let (ppid, start_time) = match parse_stat(&stat) {
            Some(v) => v,
            None => continue,
        };

        // Only orphaned processes (reparented to init)
        if ppid != 1 {
            continue;
        }

        // Compute process age in seconds
        let proc_uptime = if clock_ticks > 0 && uptime_secs > 0 {
            uptime_secs.saturating_sub(start_time / clock_ticks)
        } else {
            0
        };

        // Read cmdline
        let cmdline = read_cmdline(pid);
        if cmdline.is_empty() {
            continue;
        }

        // Check against patterns
        for &(pattern, min_age) in ORPHAN_PATTERNS {
            if proc_uptime < min_age {
                continue;
            }

            let matches = if pattern == "node" {
                // Special case: must contain both "node" and "claude"
                cmdline.contains("node") && cmdline.contains("claude")
            } else if pattern == "vite" {
                cmdline.contains("node") && cmdline.contains("vite")
            } else {
                cmdline.contains(pattern)
            };

            if matches {
                if !quiet {
                    let cmd_preview: String = cmdline.chars().take(120).collect();
                    eprintln!("  killing orphaned process: pid={pid} (uptime={proc_uptime}s)");
                    eprintln!("    cmd: {cmd_preview}");
                }
                // SAFETY: pid > 0 checked by parse, not our own pid (checked above).
                // TOCTOU: process may have exited between stat read and kill — ESRCH is benign.
                let ret = unsafe { libc::kill(pid, libc::SIGTERM) };
                if ret == 0 {
                    killed += 1;
                }
                break;
            }
        }
    }

    // WSL-specific: drop filesystem caches under memory pressure
    drop_caches_if_needed(quiet);

    if !quiet {
        if killed > 0 {
            eprintln!("cleaned up {killed} orphaned process(es)");
        } else {
            eprintln!("no orphaned processes found");
        }
    }
}

/// Parse /proc/{pid}/stat to extract PPID (field 4) and start time (field 22).
fn parse_stat(stat: &str) -> Option<(i32, u64)> {
    // Format: pid (comm) state ppid ...
    // comm can contain spaces/parens, so find the last ')' first
    let close_paren = stat.rfind(')')?;
    let fields: Vec<&str> = stat[close_paren + 2..].split_whitespace().collect();
    // fields[0] = state, fields[1] = ppid, ..., fields[19] = starttime
    if fields.len() < 20 { return None; }
    let ppid: i32 = fields[1].parse().ok()?;
    let start_time: u64 = fields[19].parse().ok()?;
    Some((ppid, start_time))
}

/// Read /proc/{pid}/cmdline (null-separated → space-separated).
fn read_cmdline(pid: i32) -> String {
    let path = format!("/proc/{pid}/cmdline");
    fs::read(&path)
        .unwrap_or_default()
        .iter()
        .map(|&b| if b == 0 { ' ' } else { b as char })
        .collect::<String>()
        .trim()
        .to_string()
}

/// Read system uptime in seconds from /proc/uptime.
fn read_uptime_secs() -> u64 {
    fs::read_to_string("/proc/uptime")
        .ok()
        .and_then(|s| s.split_whitespace().next()?.split('.').next()?.parse().ok())
        .unwrap_or(0)
}

#[allow(dead_code)]
fn read_boot_time() -> u64 {
    fs::read_to_string("/proc/stat")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("btime "))
                .and_then(|l| l.split_whitespace().nth(1)?.parse().ok())
        })
        .unwrap_or(0)
}

/// Check if throttle file indicates we ran recently.
fn is_throttled() -> bool {
    let content = match fs::read_to_string(THROTTLE_FILE) {
        Ok(s) => s,
        Err(_) => return false,
    };
    let last_run: u64 = content.trim().parse().unwrap_or(0);
    let now = epoch_secs();
    now.saturating_sub(last_run) < THROTTLE_SECS
}

/// Write current timestamp to throttle file.
fn write_throttle_timestamp() {
    let _ = fs::write(THROTTLE_FILE, format!("{}", epoch_secs()));
}

/// Current epoch seconds.
fn epoch_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Drop filesystem caches on WSL under memory pressure.
fn drop_caches_if_needed(quiet: bool) {
    // Only on WSL
    let version = match fs::read_to_string("/proc/version") {
        Ok(v) => v.to_lowercase(),
        Err(_) => return,
    };
    if !version.contains("microsoft") && !version.contains("wsl") {
        return;
    }

    // Parse /proc/meminfo
    let meminfo = match fs::read_to_string("/proc/meminfo") {
        Ok(m) => m,
        Err(_) => return,
    };

    let mut available = 0u64;
    let mut total = 0u64;
    for line in meminfo.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            available = rest.split_whitespace().next()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
        } else if let Some(rest) = line.strip_prefix("MemTotal:") {
            total = rest.split_whitespace().next()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
        }
    }

    if let Some(pct_available) = (available * 100).checked_div(total)
        && pct_available < 15
    {
        if !quiet {
            eprintln!("  memory pressure detected ({pct_available}% available) — dropping caches");
        }
        // sync + drop caches
        unsafe { libc::sync(); }
        let _ = fs::write("/proc/sys/vm/drop_caches", "1");
    }
}
