"""
Lightweight Zoopla search via Piloterr.

Usage:
    from providers.zoopla_piloterr import PiloterrZooplaClient
    client = PiloterrZooplaClient()  # reads key from Streamlit Secrets or env
    items = client.search("Flat 5, Argosy Court FY3 7NF")

Returns a list[dict] of minimal listing data (title, url, price, address, agent).
If anything fails (no key, network issues, unexpected response), it returns [].
"""

from __future__ import annotations
import os
import json
from typing import Any, Dict, List, Optional

import requests

# Streamlit is optional here — we read secrets if available
try:
    import streamlit as st
except Exception:
    st = None


class PiloterrZooplaClient:
    """
    Simple wrapper for Piloterr Zoopla endpoint.

    Looks for API key in this order:
      1) Explicit api_key argument
      2) Streamlit secrets: st.secrets["PILOTERR_API_KEY"]
      3) Environment variable: PILOTERR_API_KEY

    Endpoint override (optional):
      - Streamlit secrets: st.secrets["PILOTERR_ZOOPLA_SEARCH_ENDPOINT"]
      - Env var: PILOTERR_ZOOPLA_SEARCH_ENDPOINT
      - Default: https://www.piloterr.com/api/v2/zoopla/search
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        timeout: int = 8,
    ) -> None:
        self.api_key = (
            api_key
            or (st.secrets.get("PILOTERR_API_KEY") if st and "PILOTERR_API_KEY" in st.secrets else None)
            or os.getenv("PILOTERR_API_KEY")
        )

        self.endpoint = (
            endpoint
            or (st.secrets.get("PILOTERR_ZOOPLA_SEARCH_ENDPOINT") if st and "PILOTERR_ZOOPLA_SEARCH_ENDPOINT" in st.secrets else None)
            or os.getenv("PILOTERR_ZOOPLA_SEARCH_ENDPOINT")
            or "https://www.piloterr.com/api/v2/zoopla/search"
        )

        self.timeout = timeout

    # ---- Public API ---------------------------------------------------------
    def search(self, query: str, limit: int = 1) -> List[Dict[str, Any]]:
        """
        Perform a search. Returns [] on any failure.
        """
        if not self.api_key:
            # No key configured; caller can interpret this as "unknown"
            return []

        try:
            resp = requests.get(
                self.endpoint,
                params={"q": query, "limit": str(limit)},
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code != 200:
                return []

            data = self._safe_json(resp)
            items = self._extract_items(data)
            return items
        except Exception:
            return []

    # ---- Internal helpers ---------------------------------------------------
    def _headers(self) -> Dict[str, str]:
        # Piloterr typically supports either Authorization: Bearer or X-API-Key
        return {
            "Authorization": f"Bearer {self.api_key}",
            "X-API-Key": self.api_key,
            "Accept": "application/json",
            "User-Agent": "Blackpool-EPC-Listings-Spike/1.0",
        }

    @staticmethod
    def _safe_json(resp: requests.Response) -> Dict[str, Any]:
        try:
            return resp.json()
        except Exception:
            try:
                return json.loads(resp.text)
            except Exception:
                return {}

    @staticmethod
    def _extract_items(data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Piloterr responses can vary; try common shapes:
          - {"items": [...]}
          - {"data": [...]}
          - or already a list
        Each item is normalized to: {title, url, price, address, agent}
        """
        raw: Any = None
        if isinstance(data, list):
            raw = data
        elif isinstance(data, dict):
            if "items" in data and isinstance(data["items"], list):
                raw = data["items"]
            elif "data" in data and isinstance(data["data"], list):
                raw = data["data"]

        if not isinstance(raw, list):
            return []

        norm: List[Dict[str, Any]] = []
        for it in raw:
            if not isinstance(it, dict):
                continue
            norm.append(
                {
                    "title": it.get("title") or it.get("heading") or "",
                    "url": it.get("url") or it.get("link") or "",
                    "price": it.get("price") or it.get("price_text") or "",
                    "address": it.get("address") or it.get("location") or "",
                    "agent": (it.get("agent") or it.get("agent_name") or ""),
                }
            )
        return norm

