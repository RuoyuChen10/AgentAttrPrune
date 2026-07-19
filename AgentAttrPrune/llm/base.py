from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping, Sequence


Message = Mapping[str, str]


class LLM(ABC):
    """Minimal text-generation interface used by AgentAttrPrune."""

    @abstractmethod
    async def agen(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        num_comps: int = 1,
    ) -> str | list[str]:
        raise NotImplementedError

    @abstractmethod
    def gen(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        num_comps: int = 1,
    ) -> str | list[str]:
        raise NotImplementedError
