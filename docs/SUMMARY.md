# Memtrace - CSV-Schema ^& Beispiele
.
## CSV-Dateien
- memtrace.csv: t_start_us,t_end_us,bytes,dir,idle_us,abort_token,phase
- memtrace_abort.csv (aggregiert): abort_token,cause,count,retries_avg,conflict_avg,conflict_min,conflict_max,first_us,last_us
- memtrace_summary.txt: events_total,idle_total_us,bytes_h2d,bytes_d2h,aborts
.
## Beispiele
abort_token:
cargo run --example abort_token --features memtrace
.
stm_abort:
cargo run --example stm_abort --features memtrace -- --threads 4 --conflict {low^|med^|high} --duration 5 --seed 1