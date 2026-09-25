#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
lunar 日历/八字 MCP server（stdio，mcp SDK 2.x / MCPServer）。

数据源：lunar_python（https://github.com/6tail/lunar-python，MIT，纯 Python 零依赖，
`pip install lunar_python` 一条命令）。公历/农历/佛历/道历、干支/生肖/节气/节日、
彭祖百忌/每日宜忌、吉神方位/胎神/冲煞/纳音/星宿、八字/五行/十神、建除值星/黄道黑道等。

接入双模式（同一份配置复用同一个脚本）：
  - agent 模式：assistant/.mcp.json 配 command=python + args=[../tool/mcp_tools/lunar.py]
    （相对 assistant 目录解析到仓库根；command: python 自动换成 voice-asr）
  - llm 模式：tool/mcp.local.json 配 command=python + args=[./tool/mcp_tools/lunar.py]
    （相对仓库根；--tools all|mcp 时暴露 lunar_工具名）
  因此 lunar_python 必须装在 voice-asr 里（mcp 2.1.1 已有）。

用法：
  # 作为 MCP server（stdio），由 agent / llm 模式拉起
  python tool/mcp_tools/lunar.py
  # 独立自检（不起 server），打印各工具样例
  python tool/mcp_tools/lunar.py --selfcheck
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys

try:
    from mcp.server.mcpserver import MCPServer
    HAS_MCP = True
except Exception:  # mcp 未装时仍可 --selfcheck 独立自检
    HAS_MCP = False

try:
    from lunar_python import Solar
    from lunar_python.util import HolidayUtil
    HAS_LUNAR = True
except Exception as e:  # lunar_python 未装 → --selfcheck 也能提示缺库
    HAS_LUNAR = False
    _IMPORT_ERR = e

NOTE = "数据来自 lunar-python 日历库（https://github.com/6tail/lunar-python），供传统文化参考。"

_ZH_PAT = re.compile(r"[一-鿿]")


def _parse_date(date: str) -> tuple | None:
    """'YYYY-MM-DD' → (y,m,d)。坏输入返回 None。"""
    if not date:
        return None
    m = re.match(r"^\s*(\d{4})-(\d{1,2})-(\d{1,2})\s*$", str(date))
    if not m:
        return None
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if not (1 <= mo <= 12 and 1 <= d <= 31):
        return None
    return y, mo, d


def _parse_time(time: str) -> tuple | None:
    """'HH:MM' → (h, mi)。空 → (0,0)。坏输入返回 None。"""
    if not time:
        return 0, 0
    m = re.match(r"^\s*(\d{1,2}):(\d{1,2})\s*$", str(time))
    if not m:
        return None
    h, mi = int(m.group(1)), int(m.group(2))
    if not (0 <= h <= 23 and 0 <= mi <= 59):
        return None
    return h, mi


def _jq_year_table(year: int) -> dict:
    """某年 24 节气表：{节气名: 'YYYY-MM-DD HH:MM:SS'}（过滤英文别名 key、只留当年）。"""
    l = Solar.fromYmd(year, 6, 15).getLunar()  # 年中日期保证拿到当年全年节气
    tbl = l.getJieQiTable()
    out = {}
    for k, v in tbl.items():
        if not _ZH_PAT.search(k):   # 跳过 DA_XUE/DONG_ZHI 等英文别名
            continue
        s = v.toYmdHms()
        if s.startswith(str(year)):
            out[k] = s
    return out


def _calendar_dict(date: str) -> dict:
    """某日黄历全览 → 结构化 dict（服务端拼好，不给 LLM 原始对象）。"""
    ymd = _parse_date(date)
    if not ymd:
        return {"error": "日期格式应为 YYYY-MM-DD：%r" % (date,)}
    y, m, d = ymd
    try:
        solar = Solar.fromYmd(y, m, d)
    except Exception as e:
        return {"error": "非法日期 %s：%s" % (date, e)}
    lunar = solar.getLunar()

    out = {
        "note": NOTE,
        "date": date,
        "solar": {
            "week": "星期" + solar.getWeekInChinese(),
            "constellation": solar.getXingZuo() + "座",
            "festivals": solar.getFestivals(),
            "other_festivals": solar.getOtherFestivals(),
            "julian_day": solar.getJulianDay(),
        },
        "lunar": {
            "text": lunar.toFullString(),
            "month": lunar.getMonthInChinese(),
            "day": lunar.getDayInChinese(),
            "year_ganzhi": lunar.getYearInGanZhi(),
            "month_ganzhi": lunar.getMonthInGanZhi(),
            "day_ganzhi": lunar.getDayInGanZhi(),
            "shengxiao": lunar.getYearShengXiao(),
            "day_nayin": lunar.getDayNaYin(),
            "festivals": lunar.getFestivals(),
            "other_festivals": lunar.getOtherFestivals(),
        },
        "almanac": {
            "yi": lunar.getDayYi(),
            "ji": lunar.getDayJi(),
            "pengzu_gan": lunar.getPengZuGan(),
            "pengzu_zhi": lunar.getPengZuZhi(),
            "zhixing": lunar.getZhiXing(),                  # 建除十二值星
            "xiu": lunar.getXiu(), "xiu_luck": lunar.getXiuLuck(),   # 星宿 + 吉凶
            "tianshen": lunar.getDayTianShen(),            # 黄道黑道(青龙等十二神)
            "tianshen_type": lunar.getDayTianShenType(),
            "tianshen_luck": lunar.getDayTianShenLuck(),
            "jishen": lunar.getDayJiShen(),                # 吉神宜趋
            "xiongsha": lunar.getDayXiongSha(),            # 凶煞宜忌
            "chong": lunar.getDayChongDesc(),              # 冲
            "sha": lunar.getDaySha(),                      # 煞
            "tai_shen": lunar.getDayPositionTai(),         # 胎神方位
            "liu_yao": lunar.getLiuYao(),                  # 六曜
        },
        "auspicious_positions": {
            "xi": "%s(%s)" % (lunar.getDayPositionXi(), lunar.getDayPositionXiDesc()),
            "fu": "%s(%s)" % (lunar.getDayPositionFu(), lunar.getDayPositionFuDesc()),
            "cai": "%s(%s)" % (lunar.getDayPositionCai(), lunar.getDayPositionCaiDesc()),
            "yang_gui": "%s(%s)" % (lunar.getDayPositionYangGui(), lunar.getDayPositionYangGuiDesc()),
            "yin_gui": "%s(%s)" % (lunar.getDayPositionYinGui(), lunar.getDayPositionYinGuiDesc()),
        },
        "jieqi": {
            "current": lunar.getCurrentJieQi() or "",
            "next": lunar.getNextJieQi(True).getName() if hasattr(lunar.getNextJieQi(True), "getName") else str(lunar.getNextJieQi(True)),
        },
        "season": {
            "shujiu": getattr(lunar.getShuJiu(), "toString", lambda: str(lunar.getShuJiu()))() if lunar.getShuJiu() else "",
            "fu": lunar.getFu().toFullString() if lunar.getFu() else "",
        },
    }
    return out


def _bazi_dict(date: str, time: str, gender: str, da_yun: bool) -> dict:
    """某日/时八字排盘 → 结构化 dict。"""
    ymd = _parse_date(date)
    if not ymd:
        return {"error": "日期格式应为 YYYY-MM-DD：%r" % (date,)}
    hm = _parse_time(time)
    if hm is None:
        return {"error": "时间格式应为 HH:MM：%r" % (time,)}
    y, m, d = ymd
    h, mi = hm
    if h == 23:
        return {"error": "23:00-23:59 属下一日的子时，八字日柱会进下一天——"
                         "如需精确排盘请确认出生时间（时/分）后再查，或改用 23:00 之前的时刻。"}
    try:
        lunar = Solar.fromYmdHms(y, m, d, h, mi, 0).getLunar()
    except Exception as e:
        return {"error": "非法日期/时间 %s %s：%s" % (date, time, e)}
    ec = lunar.getEightChar()
    g = 1 if gender in ("男", "male", "1") else (0 if gender in ("女", "female", "0") else None)

    out = {
        "note": NOTE + " 八字排盘按出生地不涉及真太阳时，仅历法排盘。",
        "solar": "%s %02d:%02d" % (date, h, mi),
        "lunar_date": lunar.toFullString(),
        "four_pillars": {
            "year": ec.getYear(), "month": ec.getMonth(),
            "day": ec.getDay(), "time": ec.getTime(),
        },
        "wuxing": {
            "year": ec.getYearWuXing(), "month": ec.getMonthWuXing(),
            "day": ec.getDayWuXing(), "time": ec.getTimeWuXing(),
        },
        "nayin": {
            "year": ec.getYearNaYin(), "month": ec.getMonthNaYin(),
            "day": ec.getDayNaYin(), "time": ec.getTimeNaYin(),
        },
        "shishen_gan": {
            "year": ec.getYearShiShenGan(), "month": ec.getMonthShiShenGan(),
            "day": ec.getDayShiShenGan(), "time": ec.getTimeShiShenGan(),
        },
        "shishen_zhi": {
            "year": ec.getYearShiShenZhi(), "month": ec.getMonthShiShenZhi(),
            "day": ec.getDayShiShenZhi(), "time": ec.getTimeShiShenZhi(),
        },
        "xun_kong": {
            "year": ec.getYearXunKong(), "month": ec.getMonthXunKong(),
            "day": ec.getDayXunKong(), "time": ec.getTimeXunKong(),
        },
        "palace": {
            "ming_gong": ec.getMingGong(), "shen_gong": ec.getShenGong(),
            "tai_yuan": ec.getTaiYuan(),
        },
    }
    if da_yun and g is not None:
        try:
            yun = ec.getYun(g)
            days = [d for d in yun.getDaYun()][:5]
            out["dayun"] = {
                "gender": "男" if g == 1 else "女",
                "start": "%d年%d月%d天后（阳历 %s）" % (
                    yun.getStartYear(), yun.getStartMonth(), yun.getStartDay(),
                    yun.getStartSolar().toYmd()),
                "steps": [
                    {"age": "%d岁" % d.getStartAge(), "ganzhi": d.getGanZhi(),
                     "span": "%d-%d" % (d.getStartYear(), d.getEndYear())}
                    for d in days
                ],
            }
        except Exception as e:
            out["dayun"] = {"error": str(e)}
    elif da_yun:
        out["dayun"] = {"note": "未排大运：需提供性别（男/女）"}
    return out


def _holiday_dict(date: str) -> dict:
    """某日法定节假日/调休。"""
    ymd = _parse_date(date)
    if not ymd:
        return {"error": "日期格式应为 YYYY-MM-DD：%r" % (date,)}
    h = HolidayUtil.getHoliday(date)
    if h is None:
        return {"note": NOTE, "date": date, "holiday": None,
                "result": "%s 非法定节假日" % date}
    work = h.isWork()
    return {"note": NOTE, "date": date,
            "holiday": {"name": h.getName(), "day": h.getDay(), "target": h.getTarget(),
                        "is_workday": work,
                        "desc": ("调休上班日" if work else "放假")},
            "result": "%s %s%s" % (date, h.getName(), "（调休上班）" if work else "")}


def _jieqi_dict(year: int) -> dict:
    """某年 24 节气表。"""
    if not (1900 <= year <= 2100):
        return {"error": "年份应在 1900~2100：%r" % (year,)}
    return {"note": NOTE, "year": year, "jieqi": _jq_year_table(year)}


def _festival_dict(date: str) -> dict:
    """某日公历+农历节日。"""
    ymd = _parse_date(date)
    if not ymd:
        return {"error": "日期格式应为 YYYY-MM-DD：%r" % (date,)}
    y, m, d = ymd
    try:
        solar = Solar.fromYmd(y, m, d)
    except Exception as e:
        return {"error": "非法日期 %s：%s" % (date, e)}
    lunar = solar.getLunar()
    return {"note": NOTE, "date": date,
            "solar_festivals": solar.getFestivals(),
            "solar_other_festivals": solar.getOtherFestivals(),
            "lunar_festivals": lunar.getFestivals(),
            "lunar_other_festivals": lunar.getOtherFestivals(),
            "result": (solar.getFestivals() + solar.getOtherFestivals()
                       + lunar.getFestivals() + lunar.getOtherFestivals()) or ["（无节日）"]}


# ---------------------------------------------------------------------------
# MCP 工具
# ---------------------------------------------------------------------------
if HAS_MCP:

    server = MCPServer("lunar", description="lunar 日历库：黄历/八字/节假日/节气/节日查询")

    @server.tool()
    def get_calendar(date: str = "2026-09-25") -> str:
        """查询某日黄历全览（农历/干支/生肖/宜忌/吉神方位/冲煞/星宿/值星/黄道黑道/节日）。
        Args:
            date: 日期，格式 YYYY-MM-DD（如 2026-09-25）
        Returns: 结构化 JSON 文本（中文），含公历+农历+黄历各分项，可直接引述。
        """
        return json.dumps(_calendar_dict(date), ensure_ascii=False)

    @server.tool()
    def get_bazi(date: str = "1990-12-23", time: str = "08:37",
                 gender: str = "", da_yun: bool = True) -> str:
        """八字排盘（四柱/五行/十神/纳音/旬空/命宫身宫胎元/大运）。
        Args:
            date: 出生日期 YYYY-MM-DD
            time: 出生时间 HH:MM（23:00-23:59 属下一日子时，边界敏感请确认时刻）
            gender: 性别 男/女（要排大运必填）
            da_yun: 是否附带大运排盘（需 gender）
        Returns: 结构化 JSON 文本（中文），可直接引述。
        """
        return json.dumps(_bazi_dict(date, time, gender, da_yun), ensure_ascii=False)

    @server.tool()
    def get_holiday(date: str = "2026-10-01") -> str:
        """查询某日是否法定节假日/调休。
        Args:
            date: 日期 YYYY-MM-DD
        Returns: 结构化 JSON 文本（中文），含节日名/是否调休上班。
        """
        return json.dumps(_holiday_dict(date), ensure_ascii=False)

    @server.tool()
    def get_jieqi(year: int = 2026) -> str:
        """查询某年 24 节气表。
        Args:
            year: 年份（1900~2100）
        Returns: 结构化 JSON 文本（中文），{节气名: 时刻}。
        """
        return json.dumps(_jieqi_dict(int(year)), ensure_ascii=False)

    @server.tool()
    def get_festival(date: str = "2026-09-25") -> str:
        """查询某日公历+农历节日。
        Args:
            date: 日期 YYYY-MM-DD
        Returns: 结构化 JSON 文本（中文），公历/农历节日列表。
        """
        return json.dumps(_festival_dict(date), ensure_ascii=False)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def _selfcheck():
    """不起 server，打印各工具样例（对齐 gold --fetch 习惯）。"""
    if not HAS_LUNAR:
        print("缺少 lunar_python：请先 `pip install lunar_python`（%s）" % _IMPORT_ERR)
        return 1
    print("== lunar MCP --selfcheck ==")
    for name, fn, args in [
        ("get_calendar", _calendar_dict, ("2026-09-25",)),
        ("get_holiday", _holiday_dict, ("2026-10-01",)),
        ("get_jieqi", _jieqi_dict, (2026,)),
        ("get_festival", _festival_dict, ("2026-09-25",)),
        ("get_bazi", _bazi_dict, ("1990-12-23", "08:37", "男", True)),
        ("get_calendar(坏日期)", _calendar_dict, ("2026-13-99",)),
    ]:
        print("\n---- %s ----" % name)
        print(json.dumps(fn(*args), ensure_ascii=False, indent=1)[:1200])
    return 0


def main():
    parser = argparse.ArgumentParser(description="lunar 日历/八字 MCP server / 独立自检")
    parser.add_argument("--selfcheck", action="store_true", help="独立自检打印样例（不起 server）")
    args = parser.parse_args()

    if args.selfcheck:
        sys.exit(_selfcheck())
    if not HAS_MCP:
        raise SystemExit("缺少 mcp SDK，无法起 server；仅可用 --selfcheck。请先 pip install mcp")
    if not HAS_LUNAR:
        raise SystemExit("缺少 lunar_python，无法起 server；请先 pip install lunar_python")
    asyncio.run(server.run_stdio_async())


if __name__ == "__main__":
    main()
