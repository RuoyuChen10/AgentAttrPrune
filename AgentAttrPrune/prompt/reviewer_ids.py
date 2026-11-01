"""Run-persistent opaque source labels for final reviewers."""

from __future__ import annotations

from typing import Mapping, Sequence

import shortuuid


class AnonymousReviewerIds:
    REVIEWER_AGENT_ID_LENGTH = 4

    @classmethod
    def prepare_reviewer_agent_ids(
        cls,
        agent_ids: Sequence[str],
        reviewer_agent_ids: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        """Create or validate AgentDropout-style opaque reviewer source IDs."""

        ordered_agent_ids = tuple(str(agent_id) for agent_id in agent_ids)
        if len(set(ordered_agent_ids)) != len(ordered_agent_ids):
            raise ValueError("worker agent_ids must be unique")

        if reviewer_agent_ids is None:
            generator = shortuuid.ShortUUID()
            aliases: dict[str, str] = {}
            used: set[str] = set()
            for agent_id in ordered_agent_ids:
                alias = generator.random(length=cls.REVIEWER_AGENT_ID_LENGTH)
                while alias in used:
                    alias = generator.random(length=cls.REVIEWER_AGENT_ID_LENGTH)
                aliases[agent_id] = alias
                used.add(alias)
            return aliases

        aliases = {
            str(agent_id): str(alias)
            for agent_id, alias in reviewer_agent_ids.items()
        }
        if set(aliases) != set(ordered_agent_ids):
            raise ValueError(
                "reviewer_agent_ids must contain exactly the configured worker IDs"
            )
        if any(
            len(alias) != cls.REVIEWER_AGENT_ID_LENGTH
            or not alias.isascii()
            or not alias.isalnum()
            for alias in aliases.values()
        ):
            raise ValueError(
                "each reviewer agent ID must be a four-character alphanumeric string"
            )
        if len(set(aliases.values())) != len(aliases):
            raise ValueError("reviewer agent IDs must be unique")
        return {agent_id: aliases[agent_id] for agent_id in ordered_agent_ids}
