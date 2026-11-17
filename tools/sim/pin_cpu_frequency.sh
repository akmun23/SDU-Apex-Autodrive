#!/usr/bin/env bash
# Pins every CPU to its maximum frequency and disables the deep idle state
# (C2, 127 us wake-up), so a control thread never waits for a core to ramp up
# or leave a deep C-state. C1 (1 us wake-up) stays enabled: with only the
# polling state left, an idle hyperthread sibling busy-waits and takes
# execution resources from the control thread on its core. Needs root (the
# sysfs files are root-owned):
#
#     sudo tools/sim/pin_cpu_frequency.sh on
#     sudo tools/sim/pin_cpu_frequency.sh off      # powersave governor, full range
#     tools/sim/pin_cpu_frequency.sh status
#
# Each CPU's own cpuinfo_max_freq is used, so performance cores pin to their
# turbo maximum and efficiency cores to theirs. The setting is lost at reboot.
set -euo pipefail

cpu_root=/sys/devices/system/cpu
mode="${1:-status}"

status() {
    printf '%-6s %-12s %-10s %-10s %s\n' cpu governor min_khz max_khz cur_khz
    for dir in "$cpu_root"/cpu[0-9]*/cpufreq; do
        cpu="$(basename "$(dirname "$dir")")"
        printf '%-6s %-12s %-10s %-10s %s\n' "$cpu" \
            "$(<"$dir/scaling_governor")" "$(<"$dir/scaling_min_freq")" \
            "$(<"$dir/scaling_max_freq")" "$(<"$dir/scaling_cur_freq")"
    done
    echo "turbo disabled: $(<"$cpu_root/intel_pstate/no_turbo")"
    cpupower idle-info 2>/dev/null | grep -E "Number of idle states|Available idle states" || true
}

require_root() {
    [[ "$(id -u)" -eq 0 ]] || { echo "run with sudo: $0 $mode" >&2; exit 1; }
}

case "$mode" in
    on)
        require_root
        for dir in "$cpu_root"/cpu[0-9]*/cpufreq; do
            maximum="$(<"$dir/cpuinfo_max_freq")"
            echo "$maximum" > "$dir/scaling_max_freq"
            echo "$maximum" > "$dir/scaling_min_freq"
            echo performance > "$dir/scaling_governor"
        done
        cpupower idle-set --enable-all >/dev/null 2>&1
        cpupower idle-set --disable-by-latency 10 >/dev/null 2>&1 || \
            echo "note: idle states not changed (cpupower idle-set unavailable)" >&2
        status
        ;;
    off)
        require_root
        for dir in "$cpu_root"/cpu[0-9]*/cpufreq; do
            echo powersave > "$dir/scaling_governor"
            echo "$(<"$dir/cpuinfo_min_freq")" > "$dir/scaling_min_freq"
            echo "$(<"$dir/cpuinfo_max_freq")" > "$dir/scaling_max_freq"
        done
        cpupower idle-set --enable-all >/dev/null 2>&1 || true
        status
        ;;
    status) status ;;
    *) echo "usage: $0 on|off|status" >&2; exit 2 ;;
esac
