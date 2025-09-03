import os
import time
from typing import Dict, Any, List

import boto3
from botocore.config import Config
from fastapi import APIRouter, Query

router = APIRouter()

MODELS_CACHE: Dict[str, Any] = {"data": None, "ts": 0}
CACHE_TTL_SECONDS = 300


def _now() -> int:
    return int(time.time())


def _bedrock_client():
    """Return a boto3 Bedrock client."""
    region = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
    return boto3.client(
        "bedrock",
        region_name=region,
        config=Config(retries={"max_attempts": 5, "mode": "standard"}),
    )


def _list_foundation_models(client) -> List[Dict[str, Any]]:
    """List Bedrock foundation models and normalize for /models output."""
    out: List[Dict[str, Any]] = []
    resp = client.list_foundation_models()
    for m in resp.get("modelSummaries", []):
        mid = m.get("modelId")
        if not mid:
            continue
        out.append(
            {
                "id": mid,
                "created": _now(),
                "object": "model",
                "owned_by": "bedrock",
            }
        )
    return out


def _list_cross_region_profiles(client) -> List[Dict[str, Any]]:
    """List Bedrock system-defined inference profiles (cross-region)."""
    out: List[Dict[str, Any]] = []
    paginator = client.get_paginator("list_inference_profiles")
    for page in paginator.paginate(typeEquals="SYSTEM_DEFINED", maxResults=1000):
        for s in page.get("inferenceProfileSummaries", []):
            pid = s.get("inferenceProfileId")
            arn = s.get("inferenceProfileArn")
            created_at = s.get("createdAt")
            created_ts = (
                int(created_at.timestamp())
                if hasattr(created_at, "timestamp")
                else _now()
            )
            if pid:
                out.append(
                    {
                        "id": pid,
                        "created": created_ts,
                        "object": "model",
                        "owned_by": "bedrock",
                    }
                )
            if arn:
                out.append(
                    {
                        "id": arn,
                        "created": created_ts,
                        "object": "model",
                        "owned_by": "bedrock",
                    }
                )
    return out


def _build_models_payload(force_refresh: bool = False) -> Dict[str, Any]:
    """Build the /models payload with caching."""
    if (
        not force_refresh
        and MODELS_CACHE["data"]
        and _now() - MODELS_CACHE["ts"] < CACHE_TTL_SECONDS
    ):
        return MODELS_CACHE["data"]

    client = _bedrock_client()
    fm_entries = _list_foundation_models(client)
    xreg_entries = _list_cross_region_profiles(client)

    seen = set()
    data: List[Dict[str, Any]] = []
    for item in fm_entries + xreg_entries:
        if item["id"] in seen:
            continue
        seen.add(item["id"])
        data.append(item)

    payload = {"object": "list", "data": data}
    MODELS_CACHE["data"] = payload
    MODELS_CACHE["ts"] = _now()
    return payload


@router.get("/models")
def list_models(
    refresh: bool = Query(
        False, description="Refresh the cached model list (FMs + cross-region profiles)"
    ),
):
    return _build_models_payload(force_refresh=refresh)
