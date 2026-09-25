# -*- coding: utf-8 -*-
"""MCP stdio server：作者信息查询（mcp SDK 2.x / MCPServer）。

从另一台设备的 mcp 1.x（FastMCP）版本迁移到 2.x（本机 voice-asr 装的是 mcp 2.1.1）：
- `from mcp.server.fastmcp import FastMCP` → `from mcp.server.mcpserver import MCPServer`
- `@mcp.tool()` → `@server.add_tool`
- `mcp.run(transport='stdio')` → `asyncio.run(server.run_stdio_async())`

接入 llm+tools 模式：配置在 tool/mcp.local.json（stdio），桥接层 tool/mcp_bridge.py 自动把
本 server 的 `get_author_info` 枚举成 `<author_info_get_author_info>` 工具，`--tools mcp`
或 `--tools all` 时模型可调。
"""
from __future__ import annotations

import asyncio
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from pydantic import Field

server = MCPServer("author_info", log_level="ERROR")

# 关键年份对应的经历（数据源在这，新增年份加一行即可；未知年份返回基本信息）
_EXPERIENCE = {
    "2022": "2022年muggledy大学毕业",
    "2024": "2024年muggledy买车",
}


@server.add_tool
def get_author_info(
    year: Annotated[str, Field(
        description="年份，2022~2026（已知：2022 毕业 / 2024 买车）；不传或未知年份返回作者基本信息")] = "",
) -> str:
    """查询作者（muggledy）的经历信息。"""
    year = str(year).strip()
    if year in _EXPERIENCE:
        return _EXPERIENCE[year]
    return "作者名为muggledy"


if __name__ == "__main__":
    asyncio.run(server.run_stdio_async())
