"""Fixed, prescribed and adaptive forward integration; no scheme-name dispatch."""
from time import perf_counter
import numpy as np
from .core import History, Result, IntegrationError, Assessment


def integrate(model, scheme, initial, t_span, *, dt=None, steps=None,
              adaptive=None, initial_dt=None, min_dt=None, max_dt=None,
              snapshots=None, startup=None, diagnostics=None,
              progress=None, log_interval=30.0, max_steps=None):
    """One driver with bounded physical-state history and real-node snapshots.

    Exactly one of dt, steps, adaptive is required. Fixed/prescribed grids must
    end at t_span[1]; no silent overshoot. A final adaptive remainder may be below
    min_dt: the endpoint acts as the effective lower bound for that last step.
    Startup generates initial multistep history without embedded acceptance tests;
    it is counted separately. Supply a custom startup when studying that choice.
    """
    if sum(value is not None for value in (dt, steps, adaptive)) != 1:
        raise ValueError("Choose exactly one of dt, steps, adaptive")
    t0, end = map(float, t_span)
    if not np.isfinite([t0, end]).all() or end <= t0:
        raise ValueError("Require finite T>T0")
    if not np.isfinite(log_interval) or log_interval <= 0:
        raise ValueError("log_interval must be positive")
    if not isinstance(scheme.history_size, int) or scheme.history_size < 1:
        raise ValueError("A scheme must declare positive integer history_size")
    tolerance = 32*np.finfo(float).eps*max(abs(t0), abs(end), end-t0)
    mode = "adaptive" if adaptive is not None else ("fixed" if dt is not None else "prescribed")
    if mode != "fixed" and not scheme.variable_step:
        raise ValueError("The selected scheme does not declare variable-step support")
    if mode != "adaptive" and any(v is not None for v in (initial_dt, min_dt, max_dt)):
        raise ValueError("Adaptive step settings cannot modify a fixed/prescribed grid")
    for value in (min_dt, max_dt):
        if value is not None and (not np.isfinite(value) or value <= 0):
            raise ValueError("Step bounds must be positive and finite, or None")
    if min_dt is not None and max_dt is not None and min_dt > max_dt:
        raise ValueError("min_dt must not exceed max_dt")
    if max_steps is not None and (int(max_steps) != max_steps or max_steps < 1):
        raise ValueError("max_steps must be a positive integer or None")
    if mode == "fixed":
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")
        count = int(round((end-t0)/dt))
        if count < 1 or abs(t0+count*dt-end) > tolerance:
            raise ValueError("Fixed dt must divide the integration interval")
        prescribed = np.full(count, float(dt))
        grid = t0 + float(dt)*np.arange(count+1)
    elif mode == "prescribed":
        prescribed = np.asarray(steps, dtype=float)
        if prescribed.ndim != 1 or not len(prescribed) or not np.isfinite(prescribed).all() or np.any(prescribed <= 0):
            raise ValueError("steps must be a nonempty finite positive sequence")
        grid = np.r_[t0, t0+np.cumsum(prescribed)]
        if abs(grid[-1]-end) > tolerance:
            raise ValueError("The prescribed steps must sum to T-T0")
    else:
        if initial_dt is None or not np.isfinite(initial_dt) or initial_dt <= 0:
            raise ValueError("Adaptive integration requires a positive finite initial_dt")
        grid = None
    if grid is not None and np.any(np.diff(grid) <= 0):
        raise ValueError("Time cannot advance on this floating-point grid")
    requests = np.array([end] if snapshots is None else snapshots, dtype=float)
    if requests.ndim != 1 or not np.isfinite(requests).all() or np.any(requests < t0) or np.any(requests > end):
        raise ValueError("Snapshots must be finite times within t_span")
    selection_times = requests.copy()
    if grid is not None and len(requests):
        right = np.clip(np.searchsorted(grid, requests), 0, len(grid)-1)
        left = np.maximum(right-1, 0)
        distances = np.minimum(abs(grid[right]-requests), abs(grid[left]-requests))
        if np.any(distances > tolerance):
            raise ValueError("Fixed/prescribed snapshots must lie on the actual grid")
        selection_times = grid[np.where(abs(grid[left]-requests) <= abs(grid[right]-requests), left, right)]

    state = scheme.initialize(initial)
    if not state.finite():
        raise ValueError("Initial state must be finite")
    history = History((state,), (t0,))
    if adaptive is not None:
        adaptive.validate(scheme, state)
        adaptive.reset()
    observer = diagnostics if diagnostics is not None else getattr(model, "diagnostics", lambda s, t: {})
    starter = startup or scheme.startup
    times, accepted_dt = [t0], []
    aux = {name: [value] for name, value in state.aux.items()}
    records = [dict(observer(state, t0))]
    request_order = np.argsort(requests, kind="stable")
    mapping = np.full(len(requests), -1, dtype=int)
    selected, snapshot_states, snapshot_times = {}, [], []
    cursor = 0
    stats = {"status": "running", "accepted_steps": 0, "rejected_steps": 0,
             "forced_accept_steps": 0, "startup_steps": 0, "attempts": 0}
    began, last_progress = perf_counter(), -float("inf")
    failed_state = None

    def record_snapshots(previous, left_t, left_index, current, right_t, right_index):
        nonlocal cursor
        while cursor < len(request_order) and selection_times[request_order[cursor]] <= right_t:
            request_index = request_order[cursor]
            target = selection_times[request_index]
            choose_left = target <= left_t + (right_t-left_t)/2
            node, value, time = ((left_index, previous, left_t) if choose_left
                                 else (right_index, current, right_t))
            if node not in selected:
                selected[node] = len(snapshot_states)
                snapshot_states.append(value)
                snapshot_times.append(time)
            mapping[request_index] = selected[node]
            cursor += 1

    def notify(force=False):
        nonlocal last_progress
        now = perf_counter()
        if progress is not None and (force or now-last_progress >= log_interval):
            progress({**stats, "time": times[-1], "end": end, "elapsed_wall": now-began})
            last_progress = now

    def result():
        keys = set().union(*(record.keys() for record in records))
        diagnostics_arrays = {key: np.array([record.get(key, np.nan) for record in records]) for key in sorted(keys)}
        snapshots_arrays = {name: (np.stack([s.fields[name] for s in snapshot_states]) if snapshot_states
                                   else np.empty((0, *array.shape), dtype=array.dtype))
                            for name, array in state.fields.items()}
        return Result(np.array(times), np.array(accepted_dt), {k: np.array(v) for k, v in aux.items()},
                      diagnostics_arrays, requests, np.array(snapshot_times), mapping.copy(), snapshots_arrays,
                      history.state, {**stats, "elapsed_wall": perf_counter()-began},
                      {"scheme": scheme.name, "mode": mode, "min_dt": min_dt, "max_dt": max_dt,
                       "snapshot_policy": "nearest_accepted_node_ties_earlier" if adaptive else "strict_grid",
                       "startup": getattr(starter, "__qualname__", type(starter).__name__)}, failed_state)

    record_snapshots(state, t0, 0, state, t0, 0)
    proposal = initial_dt
    try:
        notify(force=True)
        while history.time < end:
            index = len(accepted_dt)
            if max_steps is not None and index >= max_steps:
                raise RuntimeError("User-specified max_steps reached")
            remaining = end-history.time
            if mode == "adaptive":
                trial_dt = proposal
                if min_dt is not None:
                    trial_dt = max(trial_dt, min_dt)
                if max_dt is not None:
                    trial_dt = min(trial_dt, max_dt)
                trial_dt = min(trial_dt, remaining)
            else:
                if index >= len(prescribed):
                    break
                trial_dt = float(prescribed[index])
            if not np.isfinite(trial_dt) or trial_dt <= 0 or history.time+trial_dt <= history.time:
                raise FloatingPointError("The proposed step cannot advance time")
            is_startup = len(history.states) < scheme.history_size
            stats["attempts"] += 1
            if is_startup:
                assessment = Assessment(starter(model, history, trial_dt), True, trial_dt, {"startup": 1.0})
            elif adaptive is not None:
                assessment = adaptive.attempt(model, scheme, history, trial_dt)
            else:
                assessment = Assessment(scheme.step(model, history, trial_dt), True, trial_dt)
            candidate = assessment.trial.state
            if not state.same_layout(candidate):
                raise ValueError("A trial changed the physical/auxiliary state layout")
            if not candidate.finite():
                failed_state = candidate
                raise FloatingPointError("Nonfinite trial; it cannot be forced accepted")
            if not all(np.isscalar(v) and np.isfinite(v) for v in assessment.metrics.values()):
                raise FloatingPointError("Nonfinite or nonscalar control metric")
            if not np.isfinite(assessment.next_dt) or assessment.next_dt <= 0:
                raise FloatingPointError("Controller proposed an invalid step")
            at_floor = min_dt is not None and trial_dt <= min(min_dt, remaining)*(1+16*np.finfo(float).eps)
            forced = not assessment.accept and at_floor
            if assessment.accept or forced:
                previous, previous_time = history.state, history.time
                new_time = (min(end, previous_time+trial_dt) if adaptive is not None else float(grid[index+1]))
                history = history.commit(candidate, new_time, trial_dt, scheme.history_size)
                times.append(new_time)
                accepted_dt.append(trial_dt)
                for name, value in candidate.aux.items():
                    aux[name].append(value)
                stats["accepted_steps"] += 1
                stats["startup_steps"] += int(is_startup)
                stats["forced_accept_steps"] += int(forced)
                records.append({"forced_accept": float(forced)})
                records[-1].update(dict(observer(candidate, new_time)))
                records[-1].update({"control/"+k: v for k, v in assessment.metrics.items()})
                record_snapshots(previous, previous_time, index, candidate, new_time, index+1)
                if adaptive is not None and not is_startup:
                    adaptive.on_accept(assessment, forced)
                proposal = assessment.next_dt
                notify()
                if grid is not None and index+1 == len(prescribed):
                    break
            else:
                stats["rejected_steps"] += 1
                adaptive.on_reject(assessment)
                proposal = assessment.next_dt
                if min_dt is not None:
                    proposal = max(proposal, min_dt)
                if max_dt is not None:
                    proposal = min(proposal, max_dt)
                if proposal >= trial_dt:
                    raise RuntimeError("A rejected attempt must propose a smaller step")
                notify()
        if cursor != len(request_order):
            raise RuntimeError("Some snapshot requests remain unresolved")
        stats["status"] = "completed"
        notify(force=True)
        return result()
    except BaseException as exc:
        stats.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        partial = result()
        # Preserve the original exception as cause and the valid accepted prefix.
        raise IntegrationError(str(exc), partial) from exc
