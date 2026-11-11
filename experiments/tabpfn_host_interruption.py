"""Recognize an actual host-suspension clock failure, never a model/memory failure."""


def is_host_sleep_interruption(supervision,stderr):
    if supervision.get('passed') or supervision.get('exit_code')==0:
        return False
    if 'OutOfMemory' in stderr or 'out of memory' in stderr.lower():
        return False
    if supervision.get('peak_owned_rss_plus_full_driver_bytes',6*1024**3+1)>6*1024**3:
        return False
    elapsed=supervision.get('elapsed',{})
    utc,active=elapsed.get('utc_ns'),elapsed.get('monotonic_ns')
    if type(utc) is not int or type(active) is not int or active<0 or utc-active<60*1_000_000_000:
        return False
    abort=supervision.get('aborted') or ''
    clock_abort=abort in ('operation_clock_invalid_or_over_900seconds','worker_clock_invalid_or_over_3600seconds')
    child_clock_error='Complete synchronized operation exceeded its extended dual-clock limit' in stderr
    return bool(clock_abort or (abort=='' and child_clock_error))
