"""Trace a supplied original Graph implementation without replacing its scheduler.

The factory accepts the caller's Graph class, including a class imported from a
frozen experiment snapshot. This module deliberately imports no graph classes.
Only inference with saved node dropouts is supported. Worker wrappers exist only
while ``arun`` is active and are removed even if inference raises or is cancelled.
"""

from __future__ import annotations


def traced_graph_class(base_class):
    """Return a base_class subclass recording real FIFO worker attempts/successes.

    ``execution_trace`` is replaced on each call. ``scheduled_indices`` derives
    the FIFO topological traversal from the realized edges; ``attempted_indices``
    and ``executed_indices`` record actual worker calls, including retries.
    """

    class TracedOriginalGraph(base_class):
        def _capture_round_trace(self, round_index):
            if not getattr(self, "_trace_active", False):
                return
            positions = {node_id: index for index, node_id in enumerate(self.nodes)}
            degrees = {node_id: len(node.spatial_predecessors) for node_id, node in self.nodes.items()}
            ready = [node_id for node_id, degree in degrees.items() if degree == 0]
            scheduled = []
            while ready:
                node_id = ready.pop(0)
                scheduled.append(positions[node_id])
                for successor in self.nodes[node_id].spatial_successors:
                    if successor.id not in degrees:
                        continue
                    degrees[successor.id] -= 1
                    if degrees[successor.id] == 0:
                        ready.append(successor.id)
            dropped = int(self.skip_nodes[round_index]) if self.skip_nodes else None
            self.execution_trace.append({
                "round": round_index,
                "scheduled_indices": scheduled,
                "skipped_indices": [index for index in scheduled if index == dropped],
                "attempted_indices": [],
                "executed_indices": [],
                "failed_indices": [],
                "attempt_errors": [],
                "spatial_edges": [
                    [positions[node_id], positions[target.id]]
                    for node_id, node in self.nodes.items()
                    for target in node.spatial_successors if target.id in positions
                ],
                "temporal_edges": [
                    [positions[node_id], positions[target.id]]
                    for node_id, node in self.nodes.items()
                    for target in node.temporal_successors if target.id in positions
                ],
            })

        def construct_temporal_connection(self, round=0, temperature=1.0, threshold=None):
            result = super().construct_temporal_connection(round, temperature, threshold)
            self._capture_round_trace(round)
            return result

        def construct_temporal_connection_diff(self, round=0, temperature=1.0, threshold=None):
            result = super().construct_temporal_connection_diff(round, temperature, threshold)
            self._capture_round_trace(round)
            return result

        async def arun(self, input, num_rounds=3, max_tries=3, max_time=6000, skip=False, case=False):
            if skip:
                raise ValueError("TracedOriginalGraph supports inference only; skip=True training is unsupported")
            if max_tries < 1:
                raise ValueError("max_tries must be positive")
            if getattr(self, "_trace_active", False):
                raise RuntimeError("Concurrent arun calls on the same graph are unsupported")
            self.execution_trace = []
            self._trace_active = True
            missing = object()
            originals = []
            failures = {}

            def wrap(node, index, execute):
                async def traced_execute(*args, **kwargs):
                    if not self.execution_trace:
                        raise RuntimeError("Worker executed before graph round construction")
                    trace = self.execution_trace[-1]
                    key = (trace["round"], index)
                    trace["attempted_indices"].append(index)
                    node.outputs = []
                    try:
                        result = await execute(*args, **kwargs)
                    except BaseException as error:
                        # Also record wait_for cancellation so exhausted timeouts
                        # cannot disappear when the original scheduler catches them.
                        node.outputs = []
                        failures[key] = error
                        if index not in trace["failed_indices"]:
                            trace["failed_indices"].append(index)
                        trace["attempt_errors"].append({
                            "index": index, "type": type(error).__name__, "message": str(error),
                        })
                        raise
                    failures.pop(key, None)
                    if index in trace["failed_indices"]:
                        trace["failed_indices"].remove(index)
                    trace["executed_indices"].append(index)
                    return result
                return traced_execute

            try:
                for index, node in enumerate(self.nodes.values()):
                    previous = node.__dict__.get("async_execute", missing)
                    originals.append((node, previous))
                    node.async_execute = wrap(node, index, node.async_execute)
                result = await super().arun(
                    input, num_rounds=num_rounds, max_tries=max_tries,
                    max_time=max_time, skip=False, case=case,
                )
                if failures:
                    (round_index, index), cause = next(iter(failures.items()))
                    raise RuntimeError(
                        f"Worker {index} exhausted retries in round {round_index}; inference result rejected"
                    ) from cause
                for trace in self.execution_trace:
                    expected = [index for index in trace["scheduled_indices"] if index not in trace["skipped_indices"]]
                    if trace["executed_indices"] != expected:
                        raise RuntimeError(f"Observed worker execution differs from FIFO traversal in round {trace['round']}")
                return result
            finally:
                for node, previous in originals:
                    if previous is missing:
                        del node.async_execute
                    else:
                        node.async_execute = previous
                self._trace_active = False

    TracedOriginalGraph.__name__ = f"Traced{base_class.__name__}"
    return TracedOriginalGraph
