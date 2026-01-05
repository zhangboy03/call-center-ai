#!/usr/bin/env python3
"""
Utility to delete call-state documents for a given phone number.

Usage:
    uv run python scripts/delete_calls.py --phone +8615500055988

Requires Azure CLI login or environment credentials (DefaultAzureCredential).
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

import yaml
from azure.cosmos.aio import CosmosClient
from azure.identity.aio import DefaultAzureCredential


def _load_cosmos_config() -> tuple[str, str, str]:
    config_path = Path(__file__).resolve().parents[1] / "config.yaml"
    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    cosmos_cfg = data["database"]["cosmos_db"]
    return (
        cosmos_cfg["endpoint"],
        cosmos_cfg["database"],
        cosmos_cfg["container"],
    )


async def _delete_calls(phone_number: str) -> None:
    endpoint, database, container_name = _load_cosmos_config()
    credential = DefaultAzureCredential()
    client = CosmosClient(endpoint, credential=credential)
    container = client.get_database_client(database).get_container_client(container_name)

    query = """
    SELECT c.id, c.initiate.phone_number AS partition_key
    FROM c
    WHERE STRINGEQUALS(c.initiate.phone_number, @phone, true)
       OR STRINGEQUALS(c.claim.policyholder_phone, @phone, true)
    """

    print(f"Looking for calls matching {phone_number} ...")
    to_delete: list[dict[str, Any]] = []
    async for item in container.query_items(
        query=query,
        parameters=[{"name": "@phone", "value": phone_number}],
    ):
        to_delete.append(item)

    if not to_delete:
        print("No documents found.")
        await credential.close()
        await client.close()
        return

    for item in to_delete:
        call_id = item["id"]
        partition_key = item["partition_key"]
        print(f"- Deleting call {call_id} (partition {partition_key})")
        await container.delete_item(item=call_id, partition_key=partition_key)

    await credential.close()
    await client.close()
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Delete call-state entries by phone number.")
    parser.add_argument("--phone", required=True, help="Phone number in E164 format (e.g. +8615500055988)")
    args = parser.parse_args()

    asyncio.run(_delete_calls(args.phone))


if __name__ == "__main__":
    main()

