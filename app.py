"""
Celonis Suite — Combined Single-File App
Left panel toggle switches between two modes:
  ① PQL Query Assistant  — write, verify & explain PQL (230 functions, auto-verify)
  ② Celonis AI Agent     — broad platform Q&A with live Celonis doc search
"""

import os, re, sys, time
from collections import deque
import streamlit as st
from groq import Groq
import groq as groq_errors

# ══════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Celonis Suite",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════
#  OPTIONAL IMPORTS
# ══════════════════════════════════════════════════════════════════
try:
    import tiktoken
    _ENCODER = tiktoken.get_encoding("cl100k_base")
    TIKTOKEN_OK = True
except Exception:
    TIKTOKEN_OK = False

try:
    from tavily import TavilyClient
    TAVILY_OK = True
except Exception:
    TAVILY_OK = False

try:
    from loguru import logger as _loguru
    os.makedirs("logs", exist_ok=True)
    _loguru.remove()
    _loguru.add(sys.stdout, level="INFO",
                format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")
    _loguru.add("logs/app_{time:YYYY-MM-DD}.log", level="DEBUG",
                rotation="00:00", retention="14 days", compression="zip", enqueue=True)
    LOGURU_OK = True
except Exception:
    LOGURU_OK = False

def _log(level: str, msg: str):
    if LOGURU_OK:
        getattr(_loguru, level)(msg)

# ══════════════════════════════════════════════════════════════════
#  API KEYS  (from secrets only)
# ══════════════════════════════════════════════════════════════════
GROQ_KEY   = st.secrets.get("GROQ_API_KEY",  "")
TAVILY_KEY = st.secrets.get("TAVILY_API_KEY", "")

# ══════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════
_defaults = {
    "app_mode":        "pql",       # "pql" | "agent"
    "pql_messages":    [],
    "agent_messages":  [],
    "prefill":         None,
    "panel_open":      True,
    "token_stats":     {"total": 0, "prompt": 0, "response": 0, "turns": 0},
    "rl_timestamps":   deque(),
    "rl_tier":         "free",
    # PQL-specific
    "pql_complexity":  "Advanced",
    "pql_reasoning":   True,
    "pql_queries":     0,
    "pql_verified":    0,
    "pql_fixed":       0,
    # Agent-specific
    "agent_model":     "compound-beta",
    "agent_mode":      "standard",
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════
#  SHARED UTILITIES
# ══════════════════════════════════════════════════════════════════

def count_tokens(text: str) -> int:
    if not text: return 0
    if TIKTOKEN_OK: return len(_ENCODER.encode(text))
    return int(len(text.split()) * 1.3)

def record_tokens(p: int, r: int):
    st.session_state.token_stats["prompt"]   += p
    st.session_state.token_stats["response"] += r
    st.session_state.token_stats["total"]    += p + r
    st.session_state.token_stats["turns"]    += 1

def cost_estimate(tokens: int) -> str:
    c = (tokens / 1_000_000) * 0.59
    return f"~${c:.5f}" if c < 0.001 else f"~${c:.4f}"

RATE_LIMITS = {
    "free":  {"requests": 20,  "window": 3600},
    "pro":   {"requests": 100, "window": 3600},
    "admin": {"requests": 999, "window": 3600},
}

def check_rate_limit() -> tuple[bool, str]:
    tier = st.session_state.rl_tier
    cfg  = RATE_LIMITS.get(tier, RATE_LIMITS["free"])
    ts   = st.session_state.rl_timestamps
    cutoff = time.time() - cfg["window"]
    while ts and ts[0] < cutoff: ts.popleft()
    if len(ts) >= cfg["requests"]:
        retry = int(ts[0] + cfg["window"] - time.time()) + 1
        return False, (f"⏳ Rate limit ({cfg['requests']} req/hr · **{tier}**). "
                       f"Retry in **{retry//60}m {retry%60}s**.")
    ts.append(time.time())
    return True, ""

def rl_usage() -> dict:
    tier = st.session_state.rl_tier
    cfg  = RATE_LIMITS.get(tier, RATE_LIMITS["free"])
    ts   = st.session_state.rl_timestamps
    cutoff = time.time() - cfg["window"]
    while ts and ts[0] < cutoff: ts.popleft()
    used = len(ts)
    return {"tier": tier, "used": used, "limit": cfg["requests"],
            "pct": int((used / cfg["requests"]) * 100)}

def search_available() -> bool:
    return TAVILY_OK and bool(TAVILY_KEY)

def search_celonis(query: str, max_results: int = 4) -> list[dict]:
    if not search_available(): return []
    try:
        client = TavilyClient(api_key=TAVILY_KEY)
        resp   = client.search(
            query=f"Celonis {query}", search_depth="advanced",
            include_domains=["docs.celonis.com","community.celonis.com",
                             "celonis.com","academy.celonis.com"],
            max_results=max_results,
        )
        return [{"title": r.get("title",""), "url": r.get("url",""),
                 "content": r.get("content","")[:800], "score": round(r.get("score",0),3)}
                for r in resp.get("results",[])]
    except Exception as e:
        _log("error", f"Tavily: {e}")
        return []

def format_search_ctx(results: list[dict]) -> str:
    if not results: return ""
    lines = ["### 🔍 Live Celonis Documentation\n"]
    for r in results:
        lines.append(f"**[{r['title']}]({r['url']})**\n{r['content']}\n")
    return "\n".join(lines)

# ══════════════════════════════════════════════════════════════════
#  PQL KNOWLEDGE BASE  (230 functions)
# ══════════════════════════════════════════════════════════════════
COMPACT_REFS = {
    'PU_COUNT': '''Counts non-NULL rows in source per target row.
Syntax: PU_COUNT( target_table, source_table.column [, filter_expression] )
- Returns 0 (not NULL) when no matching rows — unique among PU functions
- PREFER over PU_COUNT_DISTINCT when column is already a key (much faster)
- PU_COUNT IGNORES global filters — use filter_expression for filter-aware counts
- filter_expression is a RAW boolean expression — NEVER wrap with FILTER()
- CORRECT: PU_COUNT("CASES", "ACTIVITIES"."CASE_ID", "ACTIVITIES"."ACTIVITY" = 'Approve')
- WRONG:   PU_COUNT("CASES", "ACTIVITIES"."CASE_ID", FILTER("ACTIVITIES"."ACTIVITY" = 'Approve'))''',

    'PU_SUM': '''Sums source column per target row.
Syntax: PU_SUM( target_table, source_table.column [, filter_expression] )
- Returns NULL (not 0) when no matching rows
- PU_COUNT is cheaper than PU_SUM for counting
- filter_expression is a RAW boolean expression — NEVER wrap with FILTER()
- CORRECT: PU_SUM("VENDORS", "ORDERS"."AMOUNT", "ORDERS"."STATUS" = 'Open')
- WRONG:   PU_SUM("VENDORS", "ORDERS"."AMOUNT", FILTER("ORDERS"."STATUS" = 'Open'))''',

    'PU_AVG': '''Average of source column per target row. Always returns FLOAT.
Syntax: PU_AVG( target_table, source_table.column [, filter_expression] )
- MUCH cheaper than PU_MEDIAN — prefer PU_AVG unless true median required
- filter_expression is a RAW boolean expression — NEVER wrap with FILTER()
- CORRECT: PU_AVG("VENDORS", "ORDERS"."LEAD_TIME_DAYS", "ORDERS"."STATUS" = 'Open')
- WRONG:   PU_AVG("VENDORS", "ORDERS"."LEAD_TIME_DAYS", FILTER("ORDERS"."STATUS" = 'Open'))''',

    'PU_MAX': '''Maximum of source column per target row.
Syntax: PU_MAX( target_table, source_table.column [, filter_expression] )
- Returns NULL when no matching rows
- filter_expression is a RAW boolean expression — NEVER wrap with FILTER()''',

    'PU_MIN': '''Minimum of source column per target row.
Syntax: PU_MIN( target_table, source_table.column [, filter_expression] )
- Returns NULL when no matching rows
- filter_expression is a RAW boolean expression — NEVER wrap with FILTER()''',

    'PU_FIRST': '''Returns first element of source column for each target row.
Syntax: PU_FIRST( target_table, source_table.column [, filter_expression] [, ORDER BY source_table.column [ASC|DESC]] )
- ALWAYS use explicit ORDER BY
- filter_expression is a RAW boolean expression — NEVER wrap with FILTER()
- CORRECT: PU_FIRST("CASES", "ACTIVITIES"."TIMESTAMP", "ACTIVITIES"."ACTIVITY" = 'Create', ORDER BY "ACTIVITIES"."TIMESTAMP" ASC)
- WRONG:   PU_FIRST("CASES", "ACTIVITIES"."TIMESTAMP", FILTER("ACTIVITIES"."ACTIVITY" = 'Create'), ORDER BY "ACTIVITIES"."TIMESTAMP" ASC)
- No-filter example: PU_FIRST("CASES", "ACTIVITIES"."TIMESTAMP", ORDER BY "ACTIVITIES"."TIMESTAMP" ASC)''',

    'PU_LAST': '''Returns last element of source column for each target row.
Syntax: PU_LAST( target_table, source_table.column [, filter_expression] [, ORDER BY source_table.column [ASC|DESC]] )
- ALWAYS use explicit ORDER BY
- filter_expression is a RAW boolean expression — NEVER wrap with FILTER()
- CORRECT: PU_LAST("CASES", "ACTIVITIES"."TIMESTAMP", "ACTIVITIES"."ACTIVITY" = 'Close', ORDER BY "ACTIVITIES"."TIMESTAMP" ASC)
- WRONG:   PU_LAST("CASES", "ACTIVITIES"."TIMESTAMP", FILTER("ACTIVITIES"."ACTIVITY" = 'Close'), ORDER BY "ACTIVITIES"."TIMESTAMP" ASC)''',

    'PU_MEDIAN':         'Median per target row. Expensive — use PU_AVG when possible.\nSyntax: PU_MEDIAN( target_table, source_table.column [, filter_expression] )',
    'PU_COUNT_DISTINCT': 'Distinct count per target row. USE PU_COUNT if column is already a key.\nSyntax: PU_COUNT_DISTINCT( target_table, source_table.column [, filter_expression] )',
    'PU_MODE':           'Most frequent value per target row.\nSyntax: PU_MODE( target_table, source_table.column [, filter_expression] )',
    'PU_PRODUCT':        'Product of source column per target row.\nSyntax: PU_PRODUCT( target_table, source_table.column [, filter_expression] )',
    'PU_QUANTILE':       'Quantile (0.0-1.0) per target row.\nSyntax: PU_QUANTILE( target_table, source_table.column, quantile [, filter_expression] )',
    'PU_TRIMMED_MEAN':   'Trimmed mean per target row.\nSyntax: PU_TRIMMED_MEAN( target_table, source_table.column [, lower_cutoff [, upper_cutoff]] [, filter_expression] )',
    'PU_STRING_AGG':     'Concatenates strings from source per target row.\nSyntax: PU_STRING_AGG( target_table, source_table.column, delimiter [, filter_expression] [, ORDER BY col] )',
    'PU_STDEV':          'Standard deviation (n-1) per target row.\nSyntax: PU_STDEV( target_table, source_table.column [, filter_expression] )',

    'GLOBAL': '''Isolates aggregation from the common table — prevents join multiplication.
Syntax: GLOBAL( aggregation_expression )
- ALWAYS wrap CALC_THROUGHPUT when combined with activity-level columns
- ALWAYS wrap case-level COUNT/SUM when mixed with activity columns
- Example: GLOBAL(AVG(CALC_THROUGHPUT(CASE_START TO CASE_END, REMAP_TIMESTAMPS("ACTIVITIES"."TIMESTAMP", DAYS))))''',

    'CALC_THROUGHPUT': '''Calculates throughput time per case between two event range specifiers.
Syntax: CALC_THROUGHPUT( begin TO end, REMAP_TIMESTAMPS("ACTIVITIES"."TIMESTAMP", unit) )
begin/end: CASE_START | CASE_END | FIRST_OCCURRENCE['activity'] | LAST_OCCURRENCE['activity']
Units: DAYS | HOURS | MINUTES | SECONDS | MILLISECONDS
- Returns NULL if start > end or case has only one activity
- Wrap with GLOBAL() when combined with activity-level columns
- Example: CALC_THROUGHPUT(CASE_START TO CASE_END, REMAP_TIMESTAMPS("ACTIVITIES"."TIMESTAMP", DAYS))''',

    'CALC_REWORK': '''Counts number of activities per case.
Syntax: CALC_REWORK() | CALC_REWORK( filter_expression ) | CALC_REWORK( activity_table.column )
- Returns INT column on CASE table
- Rework detection: FILTER CALC_REWORK("ACTIVITIES"."ACTIVITY" = 'Review') > 1''',

    'REMAP_TIMESTAMPS': '''Converts DATE column to integer count of time units since epoch.
Syntax: REMAP_TIMESTAMPS( activity_table.timestamp_col, unit [, calendar_specification] )
Units: DAYS | HOURS | MINUTES | SECONDS | MILLISECONDS
- Primary use: provides timestamps argument to CALC_THROUGHPUT
- Example: REMAP_TIMESTAMPS("ACTIVITIES"."TIMESTAMP", DAYS)''',

    'MATCH_ACTIVITIES': '''Flags cases containing specified activities. Order-INDEPENDENT.
Syntax: MATCH_ACTIVITIES( [STARTING node_list] [NODE node_list] [ENDING node_list] [EXCLUDING node_list] )
- Returns 1 matching / 0 non-matching
- Example: FILTER MATCH_ACTIVITIES(NODE('Approve'), NODE('Pay'), EXCLUDING('Cancel')) = 1''',

    'MATCH_PROCESS': '''Matches cases against ordered node/edge pattern. Order-SENSITIVE.
Syntax: MATCH_PROCESS( [activity_table.string_col,] node(, node)* CONNECTED BY edge(, edge)* )
- Node types: NODE | OPTIONAL | LOOP | STARTING | ENDING
- Edge types: DIRECT [A, B] = B directly follows A; EVENTUALLY [A, B] = B eventually follows A''',

    'DATEDIFF': '''Computes difference between two dates in specified unit. Returns FLOAT.
Syntax: DATEDIFF( unit, table.date1, table.date2 ) unit: ms|ss|mi|hh|dd|mm|yy
- Example: DATEDIFF('dd', "ORDERS"."CREATE_DATE", "ORDERS"."CLOSE_DATE")''',

    'ACTIVITY_LAG':  'Returns value from preceding row by offset within same case.\nSyntax: ACTIVITY_LAG( activity_table.column [, offset] )  Default offset: 1',
    'ACTIVITY_LEAD': 'Returns value from following row by offset within same case.\nSyntax: ACTIVITY_LEAD( activity_table.column [, offset] )  Default offset: 1',
    'INDEX_ACTIVITY_ORDER': 'Returns 1-based position of each activity within its case.\nSyntax: INDEX_ACTIVITY_ORDER( activity_table.column )',
    'INDEX_ACTIVITY_LOOP':  'Returns how many times an activity has already occurred at that point.\nSyntax: INDEX_ACTIVITY_LOOP( activity_table.column )\n- 0 = first occurrence, 1 = second, etc.',

    'FILTER':        'Filters result set. Multiple FILTER statements merge by AND.',
    'FILTER_TO_NULL':'Makes functions filter-aware. Prefer PU-function filter arg when possible.',
    'BIND_FILTERS':  'Pulls filter to specified table.\nSyntax: BIND_FILTERS( target_table, condition [, condition]* )',
    'BIND':          'Pulls a value to a target table. Used for 1:N:1 relationships.\nSyntax: BIND( target_table, value )',
    'LOOKUP':        'Left outer join ignoring predefined joins.\nSyntax: LOOKUP( target_table, source_col, (join_cond) )',
    'DOMAIN_TABLE':  'Creates table with all distinct combinations.\nSyntax: DOMAIN_TABLE( table.col1, table.col2, ... )',
    'CASE':          'Conditional expression.\nSyntax: CASE WHEN cond THEN val [WHEN ...] ELSE default END',
    'COALESCE':      'First non-NULL value.\nSyntax: COALESCE( col1, col2, ..., colN )',
    'ISNULL':        'Returns 1 if NULL, 0 otherwise.\nSyntax: ISNULL( table.column )',
    'GREATEST':      'Maximum value across multiple columns.\nSyntax: GREATEST( col1, col2, ..., colN )',
    'LEAST':         'Minimum value across multiple columns.\nSyntax: LEAST( col1, col2, ..., colN )',
    'COUNT_TABLE':   'Counts rows including NULLs.\nSyntax: COUNT_TABLE( table )',
    'RUNNING_SUM':   'Cumulative sum.\nSyntax: RUNNING_SUM( column [, ORDER BY (...)] [, PARTITION BY (...)] )',
    'WINDOW_AVG':    'Average over a sliding window.\nSyntax: WINDOW_AVG( table.values, lower_bound, upper_bound [, ORDER BY ...] )',
    'INDEX_ORDER':   'Integer indices from 1.\nSyntax: INDEX_ORDER( column [, ORDER BY (...)] [, PARTITION BY (...)] )',
    'ZSCORE':        'Z-score normalization.\nSyntax: ZSCORE( table.column [, PARTITION BY (...)] )',
    'VARIANT':       'Returns process variant string per case.\nSyntax: VARIANT( activity_table.string_column )',
    'STRING_AGG':    'Aggregates strings with delimiter.\nSyntax: STRING_AGG( table.column, "delim" [, ORDER BY ...] )',
    'UPPER':         'Uppercase.\nSyntax: UPPER( table.column )',
    'LOWER':         'Lowercase.\nSyntax: LOWER( table.column )',
    'CONCAT':        'Concatenates strings.\nSyntax: CONCAT( col1, ..., colN )',
    'STRING_SPLIT':  'Splits string by pattern. Zero-based index.\nSyntax: STRING_SPLIT( table.col, pattern, index )',
    'TO_STRING':     'Converts INT or DATE to STRING.\nSyntax: TO_STRING( table.col [, FORMAT("%Y-%m-%d")] )',
    'IN_LIKE':       'Pattern matching with wildcards % and _.\nSyntax: table.col IN_LIKE( "pattern%" )',
    'REMAP_VALUES':  'Maps STRING values to new values.\nSyntax: REMAP_VALUES( table.col, [old1, new1], ..., [default] )',
    'ADD_DAYS':      'Adds days to a date.\nSyntax: ADD_DAYS( table.base_col, table.days_col )',
    'HOURS_BETWEEN': 'Difference in hours. Supports calendar.\nSyntax: HOURS_BETWEEN( table.date1, table.date2 [, calendar] )',
    'MINUTES_BETWEEN':'Difference in minutes.\nSyntax: MINUTES_BETWEEN( table.date1, table.date2 [, calendar] )',
    'SECONDS_BETWEEN':'Difference in seconds.\nSyntax: SECONDS_BETWEEN( table.date1, table.date2 [, calendar] )',
    'WORKDAYS_BETWEEN':'Number of workdays between two dates.\nSyntax: WORKDAYS_BETWEEN( calendar, table.date1, table.date2 )',
    'TODAY':         'Current date.\nSyntax: TODAY( [timezone_id] )',
    'ROUND_DAY':     'Rounds date down to day.\nSyntax: ROUND_DAY( table.date_col )',
    'ROUND_WEEK':    'Rounds date down to Monday of the week.\nSyntax: ROUND_WEEK( table.date_col )',
    'ROUND_MONTH':   'Rounds date down to first day of month.\nSyntax: ROUND_MONTH( table.date_col )',
    'ROUND_QUARTER': 'Rounds date down to beginning of quarter.',
    'CONVERT_TIMEZONE':'Converts date between timezones.\nSyntax: CONVERT_TIMEZONE( table.date_col [, from_tz], to_tz )',
    'WEEKDAY_CALENDAR':'Defines which weekdays count as work days.\nSyntax: WEEKDAY_CALENDAR( MON, TUE, ... )',
    'FACTORY_CALENDAR':'Factory calendar with specific work intervals. Used with REMAP_TIMESTAMPS.',
    'ABS':           'Absolute value.\nSyntax: ABS( table.column )',
    'POWER':         'Value raised to a power. Output: FLOAT.\nSyntax: POWER( table.col, exponent )',
    'MODULO':        'Remainder of division.\nSyntax: MODULO( dividend, divisor ) or dividend % divisor',
    'ROUND':         'Rounds to specified decimal places.\nSyntax: ROUND( table.column, decimal_places )',
    'SQRT':          'Square root.\nSyntax: SQRT( table.column )',
    'COUNT':         'Counts non-NULL rows.\nSyntax: COUNT( table.column )',
    'AVG':           'Standard average per group.\nSyntax: AVG( table.column )',
    'SUM':           'Standard sum per group.\nSyntax: SUM( table.column )',
    'MAX':           'Standard maximum per group.\nSyntax: MAX( table.column )',
    'MIN':           'Standard minimum per group.\nSyntax: MIN( table.column )',
    'COUNT_DISTINCT':'Distinct count per group.\nSyntax: COUNT_DISTINCT( table.column )',
    'MEDIAN':        'Median per group.\nSyntax: MEDIAN( table.column )',
    'QUANTILE':      'Quantile per group.\nSyntax: QUANTILE( table.column, quantile )',
    'IN':            'Membership test.\nSyntax: table.col IN( "val1", "val2" )',
    'BETWEEN':       'Range check (inclusive).\nSyntax: table.col BETWEEN lower AND upper',
    'LIKE':          'Pattern matching.\nSyntax: table.col LIKE "pattern%"',
    'CURRENCY_CONVERT':    'Converts currency using a rates table.',
    'CURRENCY_CONVERT_SAP':'Converts SAP currency using TCURR/TCURF/TCURX.',
    'KMEANS':        'K-means++ clustering.\nSyntax: KMEANS( k, col1, col2 )',
    'BPMN_CONFORMS': 'Binary BPMN conformance check (1/0).\nSyntax: BPMN_CONFORMS( event_table.col, bpmn_model [, ALLOW(...)] )',
    'CONFORMANCE':   'Petri net conformance checking. Use with READABLE() for descriptions.',
    'UNIQUE_ID':     'Unique INT for each unique tuple.\nSyntax: UNIQUE_ID( table.col1, ..., table.colN )',
    'GENERATE_RANGE':'Creates a value range. Max 10,000 elements.\nSyntax: GENERATE_RANGE( step_size, range_start, range_end )',
    'LINK_PATH':     'Traverses object links in OCPM.\nSyntax: LINK_PATH( table.col [, CONSTRAINED BY (...)] )',
    'MERGE_EVENTLOG':'Merges columns from two activity tables.\nSyntax: MERGE_EVENTLOG( target_table.col, [FILTER ...] )',
    'CREATE_EVENTLOG':'Returns activity table from OCPM object perspective.\nSyntax: CREATE_EVENTLOG( lead_object, event_type_list )',
}

PANEL_DATA = {
    'Pull-Up (PU)': [
        {'name': 'PU_COUNT',          'doc': 'Count rows per target. Returns 0 not NULL. Prefer over PU_COUNT_DISTINCT for key cols.'},
        {'name': 'PU_SUM',            'doc': 'Sum source column per target row.'},
        {'name': 'PU_AVG',            'doc': 'Average per target. Much cheaper than PU_MEDIAN.'},
        {'name': 'PU_MAX',            'doc': 'Maximum per target row.'},
        {'name': 'PU_MIN',            'doc': 'Minimum per target row.'},
        {'name': 'PU_FIRST',          'doc': 'First element per target. Always use ORDER BY.'},
        {'name': 'PU_LAST',           'doc': 'Last element per target. Always use ORDER BY.'},
        {'name': 'PU_MEDIAN',         'doc': 'Median per target. Expensive — use PU_AVG when possible.'},
        {'name': 'PU_COUNT_DISTINCT', 'doc': 'Distinct count per target. Use PU_COUNT for key columns.'},
        {'name': 'PU_STRING_AGG',     'doc': 'Concatenates strings per target row.'},
        {'name': 'PU_STDEV',          'doc': 'Standard deviation per target row.'},
    ],
    'Process': [
        {'name': 'CALC_THROUGHPUT',   'doc': 'Throughput time. Wrap in GLOBAL() with activity KPIs.'},
        {'name': 'CALC_REWORK',       'doc': 'Counts activities per case. Returns INT on case table.'},
        {'name': 'GLOBAL',            'doc': 'Prevents join multiplication. Use when mixing table levels.'},
        {'name': 'REMAP_TIMESTAMPS',  'doc': 'Remaps timestamps per unit. Required for CALC_THROUGHPUT.'},
        {'name': 'MATCH_ACTIVITIES',  'doc': 'Flags cases with activities (order-independent).'},
        {'name': 'MATCH_PROCESS',     'doc': 'Matches ordered node/edge pattern (order-sensitive).'},
        {'name': 'ACTIVITY_LAG',      'doc': 'Previous row by offset within a case.'},
        {'name': 'ACTIVITY_LEAD',     'doc': 'Next row by offset within a case.'},
        {'name': 'INDEX_ACTIVITY_ORDER', 'doc': '1-based position of each activity in case.'},
        {'name': 'INDEX_ACTIVITY_LOOP',  'doc': 'How many times activity appeared before this point.'},
        {'name': 'VARIANT',           'doc': 'Process variant string per case.'},
        {'name': 'BPMN_CONFORMS',     'doc': 'Binary BPMN conformance (1/0).'},
    ],
    'DateTime': [
        {'name': 'DATEDIFF',          'doc': 'Date difference. Units: ms|ss|mi|hh|dd|mm|yy'},
        {'name': 'HOURS_BETWEEN',     'doc': 'Difference in hours. Supports calendar.'},
        {'name': 'SECONDS_BETWEEN',   'doc': 'Difference in seconds. Supports calendar.'},
        {'name': 'WORKDAYS_BETWEEN',  'doc': 'Number of workdays between dates.'},
        {'name': 'ADD_DAYS',          'doc': 'Adds days to a date.'},
        {'name': 'TODAY',             'doc': 'Current date. Syntax: TODAY([timezone])'},
        {'name': 'ROUND_DAY',         'doc': 'Rounds down to day.'},
        {'name': 'ROUND_WEEK',        'doc': 'Rounds down to Monday of the week.'},
        {'name': 'ROUND_MONTH',       'doc': 'Rounds down to first day of month.'},
        {'name': 'CONVERT_TIMEZONE',  'doc': 'Converts date between timezones.'},
        {'name': 'WEEKDAY_CALENDAR',  'doc': 'Calendar specifying work weekdays.'},
    ],
    'Filter & Logic': [
        {'name': 'FILTER',       'doc': 'Filters result set. Multiple filters merge by AND.'},
        {'name': 'BIND_FILTERS', 'doc': 'Pulls filter to specified table.'},
        {'name': 'BIND',         'doc': 'Pulls value to target table. Used for 1:N:1 relationships.'},
        {'name': 'LOOKUP',       'doc': 'Left outer join ignoring predefined joins.'},
        {'name': 'CASE',         'doc': 'CASE WHEN cond THEN val ELSE default END'},
        {'name': 'COALESCE',     'doc': 'First non-NULL value.'},
        {'name': 'ISNULL',       'doc': 'Returns 1 if NULL, 0 otherwise.'},
        {'name': 'IN',           'doc': 'Membership test. col IN("val1","val2")'},
        {'name': 'DOMAIN_TABLE', 'doc': 'All distinct combinations of columns.'},
        {'name': 'GENERATE_RANGE','doc': 'Creates a value range. Max 10,000 elements.'},
    ],
    'String & Math': [
        {'name': 'UPPER',        'doc': 'Uppercase.'},
        {'name': 'LOWER',        'doc': 'Lowercase.'},
        {'name': 'CONCAT',       'doc': 'Concatenates strings. NULL in any arg = NULL.'},
        {'name': 'STRING_SPLIT', 'doc': 'Splits string by pattern. Zero-based index.'},
        {'name': 'IN_LIKE',      'doc': 'Pattern matching with wildcards % and _.'},
        {'name': 'REMAP_VALUES', 'doc': 'Maps STRING values to new values.'},
        {'name': 'ABS',          'doc': 'Absolute value.'},
        {'name': 'POWER',        'doc': 'Raises to a power. Output: FLOAT.'},
        {'name': 'ROUND',        'doc': 'Rounds to decimal places.'},
        {'name': 'GREATEST',     'doc': 'Maximum across columns.'},
        {'name': 'LEAST',        'doc': 'Minimum across columns.'},
    ],
    'Aggregation': [
        {'name': 'COUNT',         'doc': 'Count non-NULL rows.'},
        {'name': 'SUM',           'doc': 'Sum per group.'},
        {'name': 'AVG',           'doc': 'Average per group.'},
        {'name': 'MAX',           'doc': 'Maximum per group.'},
        {'name': 'MIN',           'doc': 'Minimum per group.'},
        {'name': 'COUNT_DISTINCT','doc': 'Distinct count per group.'},
        {'name': 'MEDIAN',        'doc': 'Median per group.'},
        {'name': 'COUNT_TABLE',   'doc': 'Counts rows including NULLs.'},
        {'name': 'RUNNING_SUM',   'doc': 'Cumulative sum. Supports ORDER BY and PARTITION BY.'},
        {'name': 'WINDOW_AVG',    'doc': 'Average over a sliding window.'},
        {'name': 'ZSCORE',        'doc': 'Z-score normalization.'},
    ],
}

CATEGORY_ICONS = {
    'Pull-Up (PU)': '⬆',
    'Process':      '⚙',
    'DateTime':     '📅',
    'Filter & Logic':'🔍',
    'String & Math':'±',
    'Aggregation':  '∑',
}

# ── PQL intent detection ──
FUNCTION_NAMES = list(COMPACT_REFS.keys())
PU_FUNCTIONS   = [fn for fn in FUNCTION_NAMES if fn.startswith("PU_")]

INTENT_PATTERNS = [
    (r'per\s+(case|vendor|order|customer|supplier|group|\w+)', PU_FUNCTIONS[:8]),
    (r'(throughput|cycle.?time|lead.?time|duration|elapsed)', ['CALC_THROUGHPUT', 'REMAP_TIMESTAMPS', 'GLOBAL']),
    (r'(rework|repeat|loop|same.?activit)', ['CALC_REWORK', 'INDEX_ACTIVITY_LOOP']),
    (r'(conform|path|sequence|activit.*order)', ['MATCH_PROCESS', 'MATCH_ACTIVITIES']),
    (r'(days?\s+between|date.?diff|workday)', ['DATEDIFF', 'HOURS_BETWEEN', 'WORKDAYS_BETWEEN']),
    (r'(variant|process.?flow)', ['VARIANT', 'MATCH_PROCESS']),
    (r'(filter|where|only.*cases|exclude)', ['FILTER', 'MATCH_ACTIVITIES', 'BIND_FILTERS']),
]

NEEDS_WORD_BOUNDARY = {
    'AVG','SUM','MAX','MIN','IN','OR','AND','NOT','ADD','SUB','DIV','LOG','LEN',
    'ABS','CEIL','FLOOR','ROUND','SQRT','DAY','MONTH','YEAR','HOURS','MINUTES',
    'SECONDS','CASE','WHEN','LIKE','RANGE','STDEV','COUNT','FILTER','BIND',
    'LOOKUP','UPPER','LOWER','MEDIAN',
}

def detect_functions(text: str) -> list[str]:
    text_lower = text.lower()
    found = set()
    for fn in FUNCTION_NAMES:
        fn_lower = fn.lower()
        if fn in NEEDS_WORD_BOUNDARY:
            if re.search(r'\b' + re.escape(fn_lower) + r'\b', text_lower):
                found.add(fn)
        else:
            if fn_lower in text_lower:
                found.add(fn)
    for pattern, fns in INTENT_PATTERNS:
        if re.search(pattern, text_lower):
            found.update(fns)
    return list(found)

def build_function_context(user_query: str) -> str:
    funcs = detect_functions(user_query)
    if not funcs:
        return ""
    docs, seen = [], set()
    for fn in funcs[:20]:
        if fn in COMPACT_REFS and fn not in seen:
            seen.add(fn)
            docs.append(f"### {fn}\n{COMPACT_REFS[fn]}")
    return "\n\n".join(docs)

# ══════════════════════════════════════════════════════════════════
#  PQL SYSTEM PROMPT + VERIFICATION
# ══════════════════════════════════════════════════════════════════

COMPLEXITY_DESC = {
    'Basic':        'Simple 1-2 function queries.',
    'Intermediate': 'Multi-function queries with filters & conditions.',
    'Advanced':     'Nested PU-functions, GLOBAL(), multi-table joins.',
    'Expert':       'Production-ready: throughput, rework, automation rate, BPMN.',
}

PQL_EXAMPLES = {
    'Basic':        ['Count activities per case','Filter cases where status is open','Difference in days between two dates'],
    'Intermediate': ['Average invoice amount per vendor','Throughput time per case in days','Find cases where Approve happens before Pay'],
    'Advanced':     ['Count late deliveries per vendor (delivery > 7 days late)','Detect rework: Review activity repeating more than twice','Flag non-conforming cases using MATCH_ACTIVITIES'],
    'Expert':       ['Full KPI: throughput + rework + automation rate','Avg approval time aggregated vendor → order → line item','BPMN conformance that tolerates undesired activities'],
}

_SQL_PROHIBITION = """
## CRITICAL — PQL IS NOT SQL. NEVER WRITE SQL.
NO: SELECT  FROM  JOIN  LEFT JOIN  GROUP BY  HAVING  WITH  OVER(...)
Always use PQL syntax: PU_AVG("LFA1", ...) not SELECT AVG(...) FROM ... GROUP BY ...
"""

_FUNCTION_GUIDE = """
## Function Selection
| Goal | Use | Avoid |
|------|-----|-------|
| Throughput per case | CALC_THROUGHPUT(CASE_START TO CASE_END, REMAP_TIMESTAMPS(...)) | PU_MAX - PU_MIN |
| Count per parent row | PU_COUNT | PU_COUNT_DISTINCT on key col |
| Average | PU_AVG | PU_MEDIAN (much slower) |
| Detect rework | INDEX_ACTIVITY_LOOP > 0 | Custom logic |
| Prevent join multiply | GLOBAL( aggregation ) | — |

## CRITICAL — PU Function Filter Syntax
The filter_expression in PU functions is a RAW boolean expression — NEVER a FILTER() call.

CORRECT — filter_expression without FILTER() wrapper:
  PU_FIRST("CASES", "ACTIVITIES"."TIMESTAMP", "ACTIVITIES"."ACTIVITY" = 'Create', ORDER BY "ACTIVITIES"."TIMESTAMP" ASC)
  PU_COUNT("CASES", "ACTIVITIES"."CASE_ID", "ACTIVITIES"."ACTIVITY" = 'Approve')
  PU_SUM("VENDORS", "ORDERS"."AMOUNT", "ORDERS"."STATUS" = 'Open')

WRONG — NEVER wrap the filter in FILTER():
  PU_FIRST("CASES", "ACTIVITIES"."TIMESTAMP", FILTER("ACTIVITIES"."ACTIVITY" = 'Create'), ...)
  PU_COUNT("CASES", "ACTIVITIES"."CASE_ID", FILTER("ACTIVITIES"."ACTIVITY" = 'Approve'))

## NULL Behaviour
| Function | No matching rows |
|----------|-----------------|
| PU_COUNT | 0 |
| PU_SUM, PU_AVG, PU_MIN, PU_MAX, PU_FIRST, PU_LAST | NULL |
| CALC_THROUGHPUT | NULL if single activity or end < start |
"""

def build_pql_system_prompt(complexity: str, show_reasoning: bool) -> str:
    ALWAYS_INCLUDE = ['GLOBAL','CALC_THROUGHPUT','PU_COUNT','PU_SUM','PU_AVG',
                      'PU_FIRST','PU_LAST','FILTER','DATEDIFF','REMAP_TIMESTAMPS',
                      'CALC_REWORK','MATCH_ACTIVITIES']
    core_refs = "\n\n".join(
        f"### {fn}\n{COMPACT_REFS[fn]}"
        for fn in ALWAYS_INCLUDE if fn in COMPACT_REFS
    )
    base = f"""You are an expert Celonis PQL engineer. Write ACCURATE, OPTIMIZED, PRODUCTION-READY PQL queries.

## PQL Core Rules
- Tables and columns MUST be double-quoted: "TABLE"."COLUMN"
- String literals use single quotes: 'value'
- PQL is column-based, NOT row-based like SQL
- Multiple FILTER statements merge by logical AND
- PU-functions aggregate FROM child table TO parent table

{_SQL_PROHIBITION}
{_FUNCTION_GUIDE}

## Core PQL Functions
{core_refs}

## Complexity: {complexity}
"""
    if show_reasoning and complexity in ("Advanced", "Expert"):
        base += "\nRespond with: 1) Analysis 2) ```pql code block 3) Explanation 4) Performance notes\n"
    else:
        base += "\nRespond with: 1) ```pql code block 2) Short plain-English explanation\n"

    base += '\nWhen table/column names are unknown use:\n"CASES"."CASE_ID", "ACTIVITIES"."ACTIVITY", "ACTIVITIES"."TIMESTAMP"\n'
    return base

VERIFICATION_SYSTEM = """You are a strict Celonis PQL validator. Review PQL and fix errors.

Rules to enforce:
1. No SQL keywords (SELECT/FROM/JOIN/GROUP BY/HAVING). PQL is NOT SQL.
2. All table/column identifiers must be double-quoted: "TABLE"."COLUMN"
3. String literals use single quotes: 'value'
4. PU functions need at least 2 arguments: PU_X(target_table, source_table.column)
5. CALC_THROUGHPUT with aggregations (AVG/COUNT/SUM) → must be wrapped in GLOBAL()
6. CRITICAL: The filter_expression inside PU functions must be a RAW boolean expression.
   NEVER wrap it with FILTER(). 
   WRONG:   PU_FIRST("CASES", "ACTIVITIES"."TS", FILTER("ACTIVITIES"."ACT" = 'X'), ORDER BY ...)
   CORRECT: PU_FIRST("CASES", "ACTIVITIES"."TS", "ACTIVITIES"."ACT" = 'X', ORDER BY ...)
   This applies to ALL PU functions: PU_COUNT, PU_SUM, PU_AVG, PU_MIN, PU_MAX, PU_FIRST, PU_LAST, etc.
7. FILTER is a standalone PQL statement — it is never used as a function call inside another function.

If correct: respond exactly VALID.
If errors: return corrected ```pql block + brief bullet list of fixes."""

def extract_pql_blocks(text: str) -> list[str]:
    return re.findall(r"```pql\s*(.*?)```", text, re.S)

def verify_pql(pql_query: str, complexity: str) -> tuple[bool, str, list[str]]:
    issues = []
    SQL_KEYWORDS = [r'\bSELECT\b', r'\bFROM\b', r'\bJOIN\b', r'\bGROUP BY\b', r'\bHAVING\b']
    for kw in SQL_KEYWORDS:
        if re.search(kw, pql_query, re.IGNORECASE):
            issues.append(f"SQL keyword found: `{kw.strip()}`")
    if 'CALC_THROUGHPUT' in pql_query and 'GLOBAL(' not in pql_query:
        if re.search(r'\b(AVG|COUNT|SUM|MEDIAN)\b', pql_query):
            issues.append("CALC_THROUGHPUT mixed with aggregations — consider GLOBAL()")
    # Detect FILTER() used as a function inside a PU function (major syntax error)
    if re.search(r'PU_\w+\s*\([^)]*FILTER\s*\(', pql_query, re.IGNORECASE):
        issues.append("CRITICAL: FILTER() used as a function inside a PU function. "
                       "The filter_expression must be a raw boolean: "
                       "e.g. \"ACTIVITIES\".\"ACTIVITY\" = 'Create' not FILTER(...)")
    if not issues and complexity not in ('Advanced', 'Expert'):
        return False, pql_query, []
    try:
        verify_prompt = f"""Review this PQL:\n```pql\n{pql_query}\n```\n{f"Flagged: {issues}" if issues else ""}\nRespond VALID or corrected ```pql block + bullet fixes."""
        resp = Groq(api_key=GROQ_KEY).chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role":"system","content":VERIFICATION_SYSTEM},
                      {"role":"user","content":verify_prompt}],
            temperature=0, max_tokens=1000,
        )
        result = resp.choices[0].message.content.strip()
        if result.upper().startswith("VALID"):
            return False, pql_query, []
        match = re.search(r"```pql\s*(.*?)```", result, re.S)
        if match:
            fixes = re.findall(r'^[-•*]\s+(.+)', result, re.MULTILINE)
            return True, match.group(1).strip(), fixes or ["Corrected by verification pass"]
        return False, pql_query, []
    except Exception as e:
        return False, pql_query, [f"Verification skipped: {e}"]

# ══════════════════════════════════════════════════════════════════
#  AGENT SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════════

AGENT_MODELS = {
    "⚡ Compound (Web Search)": "compound-beta",
    "🦙 Llama 3.3 70B":         "llama-3.3-70b-versatile",
    "🦙 Llama 4 Scout 17B":     "meta-llama/llama-4-scout-17b-16e-instruct",
    "🔮 Mixtral 8x7B":          "mixtral-8x7b-32768",
}
AGENT_MODES = {
    "🎓 Guided (Beginner)":  "beginner",
    "⚡ Standard":            "standard",
    "🔬 Expert / Deep Dive": "expert",
    "📝 PQL Only":           "pql_only",
}
MODE_INSTRUCTIONS = {
    "beginner": "Use simple language, explain every term, use analogies, be encouraging. Add 'Plain English' summaries.",
    "standard": "Use ### headings, code blocks, **bold** key terms, 💡 Pro Tip, ⚠️ Common Pitfall, numbered steps.",
    "expert":   "Skip basics. Deep architecture, advanced PQL, performance, edge cases, enterprise trade-offs.",
    "pql_only": "Answer ONLY with PQL. Show complete snippets, explain each function, show alternatives, performance notes.",
}
AGENT_SUGGESTIONS = [
    ("OLAP Views",       "How do I create an OLAP view in Celonis step by step?"),
    ("PQL Basics",       "What is PQL and how do I write a basic aggregation query?"),
    ("Process Explorer", "How does Process Explorer work in Celonis?"),
    ("Data Model",       "How do I create and configure a data model in Celonis?"),
    ("SAP Connector",    "How do I connect SAP ECC or S/4HANA data to Celonis?"),
    ("KPI Trees",        "How do I build KPI trees in Celonis Studio?"),
    ("Action Flows",     "How can I set up action flows and automations in Celonis?"),
    ("ML Workbench",     "What is the ML Workbench used for in Celonis?"),
    ("Permissions",      "How does role-based access and permissions work in Celonis?"),
]

def build_agent_prompt(answer_mode: str, search_ctx: str = "") -> str:
    base = """You are an expert Celonis AI Assistant for the Celonis Process Intelligence platform.
Deep expertise: Studio, OLAP Views, Process Explorer, Data Models, PQL, Connectors (SAP ECC/S4HANA,
Salesforce, ServiceNow), ML Workbench, Permissions, Action Engine, KPI Trees.
Respond like a senior Celonis consultant — expert, practical, friendly."""
    mode_str   = MODE_INSTRUCTIONS.get(answer_mode, MODE_INSTRUCTIONS["standard"])
    search_str = f"\n\n### 🔍 Live Search Results\n{search_ctx}" if search_ctx else ""
    return f"{base}\n\n## Response Style\n{mode_str}{search_str}"

# ══════════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Sora:wght@300;400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

html,body{background:#0b0e1a !important;color:#e8ecf4 !important;}
[data-testid="stAppViewContainer"],[data-testid="stMain"],.main,.block-container{
    background:#0b0e1a !important;color:#e8ecf4 !important;
    font-family:'Sora',sans-serif !important;padding-top:0 !important;max-width:100% !important;}
#MainMenu,footer,header,[data-testid="stHeader"],
[data-testid="collapsedControl"],[data-testid="stSidebarCollapsedControl"]{display:none !important;}
[data-testid="stSidebar"]{display:none !important;}

p,span,li,div,label,h1,h2,h3,h4,h5,h6,
.stMarkdown,[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li{color:#e8ecf4 !important;}

/* inputs */
[data-testid="stTextInput"] input,[data-baseweb="select"]>div{
    background:#1a2340 !important;color:#e8ecf4 !important;
    border:1px solid #2a3a5c !important;border-radius:8px !important;
    font-family:'Sora',sans-serif !important;font-size:13px !important;}
[data-testid="stTextInput"] input::placeholder{color:#7a8aab !important;}
[data-baseweb="popover"] ul,[data-baseweb="menu"] ul{background:#1a2340 !important;}
[data-baseweb="popover"] li,[data-baseweb="menu"] li{color:#e8ecf4 !important;}
[data-baseweb="popover"] li:hover{background:#2a3a5c !important;}

/* buttons */
.stButton>button{
    background:rgba(255,255,255,0.04) !important;border:1px solid #2a3a5c !important;
    color:#c8d4e8 !important;font-family:'Sora',sans-serif !important;
    font-size:12px !important;text-align:left !important;border-radius:8px !important;
    padding:8px 12px !important;width:100% !important;white-space:normal !important;
    height:auto !important;line-height:1.5 !important;margin-bottom:3px !important;
    transition:all 0.18s !important;}
.stButton>button:hover{
    background:rgba(47,110,245,0.2) !important;border-color:#2f6ef5 !important;color:#fff !important;}

/* toggle button */
div[data-testid="column"]:first-child .stButton:first-child>button{
    width:38px !important;height:38px !important;min-width:38px !important;
    padding:0 !important;font-size:18px !important;text-align:center !important;
    border-radius:10px !important;background:#1a2340 !important;
    border:1px solid #2a3a5c !important;color:#e8ecf4 !important;
    margin-bottom:0 !important;line-height:38px !important;}

/* mode switcher pills */
.mode-switch{display:flex;gap:0;background:#111627;border:1px solid #2a3a5c;border-radius:10px;padding:3px;margin-bottom:14px;}
.mode-btn{flex:1;text-align:center;padding:8px 4px;border-radius:7px;cursor:pointer;font-size:11px;
    font-family:'Space Mono',monospace;font-weight:700;color:#7a8aab;transition:all 0.2s;}
.mode-btn.active-pql{background:linear-gradient(135deg,#2f6ef5,#1a4fba);color:#fff;
    box-shadow:0 0 12px rgba(47,110,245,0.4);}
.mode-btn.active-agent{background:linear-gradient(135deg,#ff6d29,#cc4a12);color:#fff;
    box-shadow:0 0 12px rgba(255,109,41,0.4);}

/* alerts */
.stAlert{border-radius:8px !important;background:#1a2340 !important;}
.stAlert p{color:#e8ecf4 !important;font-size:12px !important;}

/* metric */
[data-testid="stMetric"] label{color:#9ab0cc !important;font-size:11px !important;}
[data-testid="stMetricValue"]{color:#00d4b4 !important;font-size:18px !important;}

/* selectbox */
[data-testid="stSelectbox"] label{color:#c8d4e8 !important;font-size:11px !important;}
[data-baseweb="select"]>div{font-size:12px !important;}

/* slider */
[data-testid="stSlider"] .stSlider{padding:0 !important;}
[data-testid="stSlider"] label{color:#c8d4e8 !important;font-size:11px !important;}

/* toggle */
[data-testid="stToggle"] label{color:#c8d4e8 !important;font-size:11px !important;}
[data-testid="stToggle"] input:checked+div{background:#2f6ef5 !important;}

/* expanders */
[data-testid="stExpander"]{background:#111627 !important;border:1px solid #2a3a5c !important;border-radius:8px !important;margin-bottom:3px !important;}
[data-testid="stExpander"] summary{font-family:'Space Mono',monospace !important;font-size:11px !important;color:#9ab0cc !important;padding:7px 10px !important;}
[data-testid="stExpander"] summary:hover{color:#e8ecf4 !important;}

/* progress bar */
[data-testid="stProgressBar"]>div{background:#1a2340 !important;border-radius:10px !important;}
[data-testid="stProgressBar"]>div>div{background:linear-gradient(90deg,#2f6ef5,#00d4b4) !important;border-radius:10px !important;}

/* chat messages */
[data-testid="stChatMessage"]{
    background:#141b2d !important;border:1px solid #2a3a5c !important;
    border-radius:14px !important;padding:14px 18px !important;margin-bottom:10px !important;}
[data-testid="stChatMessage"] p,[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p{
    color:#e8ecf4 !important;font-size:14px !important;line-height:1.8 !important;}
[data-testid="stChatMessage"] h1,[data-testid="stChatMessage"] h2,
[data-testid="stChatMessage"] h3{color:#7dd3fc !important;font-weight:700 !important;}
[data-testid="stChatMessage"] strong{color:#c8d8ff !important;}
[data-testid="stChatMessage"] code{
    background:#1e2d4a !important;color:#00d4b4 !important;
    padding:2px 7px !important;border-radius:5px !important;
    font-family:'IBM Plex Mono',monospace !important;font-size:12.5px !important;}
[data-testid="stChatMessage"] pre{
    background:#0d1525 !important;border:1px solid #2a3a5c !important;
    border-radius:10px !important;padding:16px !important;overflow-x:auto !important;}
[data-testid="stChatMessage"] pre code{
    background:none !important;color:#a5d6ff !important;padding:0 !important;font-size:13px !important;}

/* chat input */
[data-testid="stChatInput"]>div,[data-testid="stChatInput"]{
    background:#141b2d !important;border:1px solid #2a3a5c !important;border-radius:14px !important;}
[data-testid="stChatInput"] textarea{
    background:#141b2d !important;color:#e8ecf4 !important;
    font-family:'Sora',sans-serif !important;font-size:14px !important;caret-color:#2f6ef5 !important;}
[data-testid="stChatInput"] textarea::placeholder{color:#7a8aab !important;}
[data-testid="stBottom"]{background:#0b0e1a !important;border-top:1px solid #1a2340 !important;padding:10px 0 !important;}

/* header bar */
.cel-header{
    background:linear-gradient(135deg,#141b2d,#1a2340);border:1px solid #2a3a5c;border-radius:14px;
    padding:12px 18px;margin-bottom:12px;display:flex;align-items:center;justify-content:space-between;}
.cel-title{font-size:17px;font-weight:700;color:#fff;margin:0;}
.cel-sub{font-size:11px;color:#7dd3fc;font-family:'Space Mono',monospace;margin:2px 0 0;}
.cel-badge{display:flex;align-items:center;gap:7px;font-size:11px;color:#00d4b4;
    font-family:'Space Mono',monospace;font-weight:700;background:rgba(0,212,180,0.1);
    border:1px solid rgba(0,212,180,0.3);border-radius:20px;padding:5px 14px;}
.dot{width:7px;height:7px;background:#00d4b4;border-radius:50%;
     box-shadow:0 0 7px #00d4b4;display:inline-block;animation:blink 2s infinite;}
@keyframes blink{0%,100%{opacity:1;}50%{opacity:0.3;}}

/* panel */
.panel-box{background:#111627;border:1px solid #2a3a5c;border-radius:16px;padding:18px 14px;}
.divider{height:1px;background:#2a3a5c;margin:12px 0;}
.sec-label{font-size:9px;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;
    color:#7a8aab;font-family:'Space Mono',monospace;margin-bottom:7px;}

/* welcome */
.welcome-box{text-align:center;padding:44px 20px;
    background:linear-gradient(135deg,#141b2d,#1a2340);
    border:1px solid #2a3a5c;border-radius:18px;margin:0 0 16px;}
.wel-title{font-size:22px;font-weight:700;color:#fff;margin-bottom:10px;}
.wel-title .pql-c{color:#2f6ef5;} .wel-title .ag-c{color:#ff6d29;}
.wel-sub{font-size:13px;color:#9ab0cc;max-width:520px;margin:0 auto;line-height:1.9;}
.wel-sub em{color:#7dd3fc;font-style:normal;font-weight:600;}

/* badges */
.badge-row{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px;}
.badge{border-radius:20px;padding:5px 14px;font-size:11px;
    font-family:'Space Mono',monospace;font-weight:700;display:inline-flex;align-items:center;gap:5px;}
.b-blue{background:rgba(47,110,245,0.15);border:1px solid rgba(47,110,245,0.4);color:#7dd3fc;}
.b-green{background:rgba(0,212,180,0.12);border:1px solid rgba(0,212,180,0.35);color:#00d4b4;}
.b-orange{background:rgba(255,109,41,0.12);border:1px solid rgba(255,109,41,0.35);color:#ff9a56;}
.b-red{background:rgba(239,68,68,0.12);border:1px solid rgba(239,68,68,0.35);color:#fca5a5;}

/* verify badges */
.verify-pass{display:flex;align-items:center;gap:8px;background:#052e16;border:1px solid #10b981;
    border-radius:8px;padding:8px 14px;color:#6ee7b7;font-size:12px;
    font-family:'Space Mono',monospace;margin-top:8px;}
.verify-fix{display:flex;align-items:center;gap:8px;background:#451a03;border:1px solid #f59e0b;
    border-radius:8px;padding:8px 14px;color:#fcd34d;font-size:12px;
    font-family:'Space Mono',monospace;margin-top:8px;}

::-webkit-scrollbar{width:5px;height:5px;}
::-webkit-scrollbar-track{background:#0b0e1a;}
::-webkit-scrollbar-thumb{background:#2a3a5c;border-radius:4px;}
::-webkit-scrollbar-thumb:hover{background:#2f6ef5;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  TOP BAR  (toggle + header)
# ══════════════════════════════════════════════════════════════════
toggle_col, header_col = st.columns([0.4, 9.6], gap="small")

with toggle_col:
    icon = "✕" if st.session_state.panel_open else "☰"
    if st.button(icon, key="toggle_panel"):
        st.session_state.panel_open = not st.session_state.panel_open
        st.rerun()

with header_col:
    mode_label = "PQL Query Assistant" if st.session_state.app_mode == "pql" else "Celonis AI Agent"
    mode_color = "#2f6ef5" if st.session_state.app_mode == "pql" else "#ff6d29"
    feat_badges = ""
    if search_available(): feat_badges += '<span style="font-size:11px;color:#00d4b4;font-family:Space Mono,monospace;margin-left:10px;">🔍 Live Search ON</span>'
    st.markdown(f"""
    <div class="cel-header">
      <div style="display:flex;align-items:center;gap:12px;">
        <div style="font-size:26px;">⚡</div>
        <div>
          <p class="cel-title">Celonis Suite &nbsp;
            <span style="font-size:13px;color:{mode_color};font-family:Space Mono,monospace;font-weight:700;">
              [{mode_label}]
            </span>{feat_badges}
          </p>
        </div>
      </div>
      <div class="cel-badge"><span class="dot"></span> Live</div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  MAIN LAYOUT
# ══════════════════════════════════════════════════════════════════
if st.session_state.panel_open:
    left_col, right_col = st.columns([2.3, 7.7], gap="medium")
else:
    left_col, right_col = st.columns([0.01, 9.99], gap="small")

# ──────────────────────────────────────────────────────────────────
#  LEFT PANEL
# ──────────────────────────────────────────────────────────────────
with left_col:
    if not st.session_state.panel_open:
        pass
    else:
        st.markdown('<div class="panel-box">', unsafe_allow_html=True)

        # Logo
        st.markdown("""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
          <div style="width:36px;height:36px;border-radius:10px;
                      background:linear-gradient(135deg,#ff6d29,#ff9a56);
                      display:flex;align-items:center;justify-content:center;
                      font-size:18px;box-shadow:0 0 14px rgba(255,109,41,0.4);">⚡</div>
          <div>
            <div style="font-size:15px;font-weight:700;color:#fff;">Celonis Suite</div>
            <div style="font-size:10px;color:#7a8aab;font-family:Space Mono,monospace;">Powered by Groq</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── MODE SWITCHER ──────────────────────────────────────
        st.markdown('<div class="sec-label">⚡ Switch Mode</div>', unsafe_allow_html=True)
        is_pql   = st.session_state.app_mode == "pql"
        is_agent = st.session_state.app_mode == "agent"

        col_pql, col_agent = st.columns(2)
        with col_pql:
            pql_style = "background:linear-gradient(135deg,#2f6ef5,#1a4fba);color:#fff;border:1px solid #2f6ef5;" if is_pql else "background:rgba(47,110,245,0.08);color:#7a8aab;border:1px solid #2a3a5c;"
            if st.button("📝 PQL\nAssistant", key="switch_pql", use_container_width=True):
                st.session_state.app_mode = "pql"
                st.rerun()
        with col_agent:
            if st.button("🤖 Celonis\nAI Agent", key="switch_agent", use_container_width=True):
                st.session_state.app_mode = "agent"
                st.rerun()

        st.markdown(f"""
        <div style="font-size:10px;color:#7a8aab;font-family:Space Mono,monospace;
                    text-align:center;margin:-2px 0 10px;padding:4px 8px;
                    background:{'rgba(47,110,245,0.1)' if is_pql else 'rgba(255,109,41,0.1)'};
                    border-radius:6px;border:1px solid {'#2f6ef5' if is_pql else '#ff6d29'};">
            {'📝 PQL Query Assistant' if is_pql else '🤖 Celonis AI Agent'} active
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # ── PQL-SPECIFIC PANEL ─────────────────────────────────
        if is_pql:
            st.markdown('<div class="sec-label">🎯 Complexity</div>', unsafe_allow_html=True)
            complexity = st.select_slider(
                "Complexity", options=['Basic','Intermediate','Advanced','Expert'],
                value=st.session_state.pql_complexity, label_visibility="collapsed",
            )
            st.session_state.pql_complexity = complexity
            st.caption(COMPLEXITY_DESC[complexity])

            st.session_state.pql_reasoning = st.toggle(
                "Show query reasoning", value=st.session_state.pql_reasoning,
            )

            st.markdown('<div class="sec-label">💡 Quick Examples</div>', unsafe_allow_html=True)
            for ex in PQL_EXAMPLES.get(complexity, PQL_EXAMPLES['Advanced']):
                if st.button(f"→ {ex}", key=f"pql_ex_{ex}", use_container_width=True):
                    st.session_state.prefill = ex
                    st.rerun()

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

            # Function reference
            st.markdown('<div class="sec-label">📚 Function Reference</div>', unsafe_allow_html=True)
            search_fn = st.text_input("Search functions", placeholder="Search 60+ functions…",
                                      label_visibility="collapsed", key="fn_search")
            for cat, funcs in PANEL_DATA.items():
                hits = [f for f in funcs if not search_fn
                        or search_fn.lower() in f['name'].lower()
                        or search_fn.lower() in f['doc'].lower()]
                if not hits: continue
                icon = CATEGORY_ICONS.get(cat, '•')
                with st.expander(f"{icon} {cat} ({len(hits)})"):
                    for fn in hits:
                        if st.button(fn['name'], key=f"fn_{fn['name']}_{cat}", use_container_width=True):
                            st.session_state.prefill = (
                                f"Write a PQL query using {fn['name']} and explain the syntax with a practical example."
                            )
                            st.rerun()
                        st.caption(fn['doc'][:100] + '…' if len(fn['doc']) > 100 else fn['doc'])

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

            # PQL stats
            st.markdown('<div class="sec-label">📊 PQL Stats</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Queries", st.session_state.pql_queries)
            c2.metric("✅", st.session_state.pql_verified)
            c3.metric("🔧", st.session_state.pql_fixed)

        # ── AGENT-SPECIFIC PANEL ───────────────────────────────
        else:
            st.markdown('<div class="sec-label">🤖 Model</div>', unsafe_allow_html=True)
            agent_model_label = st.selectbox(
                "Model", list(AGENT_MODELS.keys()), index=0,
                label_visibility="collapsed", key="agent_model_label",
            )
            st.session_state.agent_model = AGENT_MODELS[agent_model_label]

            st.markdown('<div class="sec-label">🎯 Answer Mode</div>', unsafe_allow_html=True)
            agent_mode_label = st.selectbox(
                "Mode", list(AGENT_MODES.keys()), index=1,
                label_visibility="collapsed", key="agent_mode_label",
            )
            st.session_state.agent_mode = AGENT_MODES[agent_mode_label]

            c1, c2 = st.columns(2)
            with c1:
                if search_available(): st.success("🔍 Search ON", icon="✅")
                else: st.warning("🔍 Off", icon="⚠️")
            with c2:
                st.success("🧮 Tiktoken" if TIKTOKEN_OK else "🧮 Approx",
                           icon="✅" if TIKTOKEN_OK else "ℹ️")

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="sec-label">💬 Quick Questions</div>', unsafe_allow_html=True)
            for tag, question in AGENT_SUGGESTIONS:
                if st.button(f"[{tag}]  {question}", key=f"ag_{tag}", use_container_width=True):
                    st.session_state.prefill = question
                    st.rerun()

        # ── SHARED BOTTOM ──────────────────────────────────────
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-label">📊 Session Stats</div>', unsafe_allow_html=True)
        stats = st.session_state.token_stats
        usage = rl_usage()
        ca, cb = st.columns(2)
        with ca: st.metric("Tokens", f"{stats['total']:,}")
        with cb: st.metric("Cost",   cost_estimate(stats["total"]))

        pct = usage["pct"]
        bar_icon = "🟢" if pct < 60 else ("🟡" if pct < 85 else "🔴")
        st.markdown(
            f'<div style="font-size:10px;color:#9ab0cc;margin-bottom:4px;">'
            f'{bar_icon} {usage["used"]}/{usage["limit"]} req/hr · {usage["tier"].upper()}</div>',
            unsafe_allow_html=True,
        )
        st.progress(min(pct / 100, 1.0))

        clear_key = "clear_pql" if is_pql else "clear_agent"
        if st.button("🗑️ Clear Chat", use_container_width=True, key=clear_key):
            if is_pql:
                st.session_state.pql_messages = []
            else:
                st.session_state.agent_messages = []
            st.session_state.token_stats = {"total":0,"prompt":0,"response":0,"turns":0}
            st.rerun()

        st.markdown(
            '<p style="font-size:9px;color:#7a8aab;text-align:center;'
            'font-family:Space Mono,monospace;margin-top:10px;">'
            'console.groq.com · app.tavily.com</p>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  RIGHT PANEL — CHAT AREA
# ══════════════════════════════════════════════════════════════════
with right_col:
    if not GROQ_KEY:
        st.error("⚠️ No Groq API key found. Add `GROQ_API_KEY` to your Streamlit secrets.")
        st.stop()

    app_mode   = st.session_state.app_mode
    messages   = st.session_state.pql_messages if app_mode == "pql" else st.session_state.agent_messages
    complexity = st.session_state.pql_complexity
    show_rsn   = st.session_state.pql_reasoning

    # ── Welcome screen ──
    if not messages:
        if app_mode == "pql":
            st.markdown("""
            <div class="welcome-box">
              <div style="font-size:48px;margin-bottom:14px;">📝</div>
              <div class="wel-title">PQL Query <span class="pql-c">Assistant</span></div>
              <div class="wel-sub">
                Write, explain, and optimize PQL queries with <em>230 functions</em> and
                an <em>automatic verification pass</em> that catches SQL mistakes and
                common PQL anti-patterns before you see the result.<br/><br/>
                <em>Try: "Avg throughput time per case" · "Detect rework loops" · "How does PU_COUNT work?"</em>
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="welcome-box">
              <div style="font-size:48px;margin-bottom:14px;">🤖</div>
              <div class="wel-title">Celonis <span class="ag-c">AI Agent</span></div>
              <div class="wel-sub">
                Expert Q&A for the entire Celonis platform — Studio, OLAP Views,
                Process Explorer, Data Models, SAP connectors, and more.<br/>
                Combines <em>live Celonis doc search</em> with AI to give accurate, step-by-step answers.<br/><br/>
                <em>Try: "How do I create an OLAP view?" · "Explain SAP connector setup" · "What is a KPI tree?"</em>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Chat history ──
    for msg in messages:
        avatar = ("📝" if app_mode == "pql" else "🤖") if msg["role"] == "assistant" else "👤"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    prefill = st.session_state.pop("prefill", None)

    placeholder_text = (
        "Describe your PQL query, ask about a function, or paste PQL to optimize…"
        if app_mode == "pql"
        else "Ask about OLAP views, PQL, data models, SAP connectors, or paste any Celonis question…"
    )
    prompt = st.chat_input(placeholder_text, key="chat_input") or prefill

    # ══════════════════════════════════════════════════════════
    #  HANDLE PROMPT
    # ══════════════════════════════════════════════════════════
    if prompt:
        allowed, rl_msg = check_rate_limit()
        if not allowed:
            st.warning(rl_msg)
            st.stop()

        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        messages.append({"role": "user", "content": prompt})
        if app_mode == "pql":
            st.session_state.pql_messages = messages
        else:
            st.session_state.agent_messages = messages

        history = [{"role": m["role"], "content": m["content"]} for m in messages[-20:]]
        avatar_ai = "📝" if app_mode == "pql" else "🤖"

        with st.chat_message("assistant", avatar=avatar_ai):
            t0 = time.time()

            # ── PQL MODE ──────────────────────────────────────
            if app_mode == "pql":
                func_ctx    = build_function_context(prompt)
                system      = build_pql_system_prompt(complexity, show_rsn)
                if func_ctx:
                    system += f"\n\n## Relevant Functions (auto-retrieved)\n{func_ctx}"

                p_tokens = count_tokens(system) + count_tokens(prompt)

                try:
                    stream = Groq(api_key=GROQ_KEY).chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role":"system","content":system}] + history,
                        max_completion_tokens=2048,
                        temperature=0.15,
                        stream=True,
                    )
                    full_response, ph = "", st.empty()
                    for chunk in stream:
                        delta = chunk.choices[0].delta.content or ""
                        full_response += delta
                        ph.markdown(full_response + "▌")
                    ph.markdown(full_response)

                    r_tokens = count_tokens(full_response)
                    record_tokens(p_tokens, r_tokens)
                    latency  = (time.time() - t0) * 1000
                    st.session_state.pql_queries += 1

                    # Verification pass
                    pql_blocks = extract_pql_blocks(full_response)
                    if pql_blocks:
                        for pql_block in pql_blocks:
                            was_fixed, final_query, fix_notes = verify_pql(pql_block, complexity)
                            if was_fixed:
                                st.session_state.pql_fixed += 1
                                st.markdown('<div class="verify-fix">🔧 <strong>Auto-corrected</strong> — verification pass fixed issues</div>', unsafe_allow_html=True)
                                for note in fix_notes:
                                    st.caption(f"  • {note}")
                                st.markdown("**Corrected query:**")
                                st.code(final_query, language="sql")
                                full_response = full_response.replace(
                                    f"```pql\n{pql_block}\n```",
                                    f"```pql\n{final_query}\n```"
                                )
                            else:
                                st.session_state.pql_verified += 1
                                st.markdown('<div class="verify-pass">✅ <strong>Verified</strong> — query passed correctness check</div>', unsafe_allow_html=True)

                    badges = f'<div class="badge-row"><span class="badge b-blue">📝 PQL Mode · {complexity}</span><span class="badge b-green">⏱ {latency/1000:.1f}s · {r_tokens} tok</span></div>'
                    st.markdown(badges, unsafe_allow_html=True)

                    messages.append({"role": "assistant", "content": full_response})
                    st.session_state.pql_messages = messages

                except groq_errors.AuthenticationError:
                    st.error("❌ Invalid Groq API Key.")
                except groq_errors.RateLimitError:
                    st.error("⏳ Groq rate limit hit — wait and retry.")
                except Exception as e:
                    st.error(f"⚠️ Error: {e}")

            # ── AGENT MODE ────────────────────────────────────
            else:
                search_results, search_ctx = [], ""
                selected_model = st.session_state.agent_model
                answer_mode    = st.session_state.agent_mode

                if search_available():
                    with st.spinner("🔍 Searching Celonis docs…"):
                        search_results = search_celonis(prompt)
                        search_ctx     = format_search_ctx(search_results)

                system   = build_agent_prompt(answer_mode, search_ctx)
                p_tokens = count_tokens(system) + count_tokens(prompt)

                try:
                    stream = Groq(api_key=GROQ_KEY).chat.completions.create(
                        model=selected_model,
                        messages=[{"role":"system","content":system}] + history,
                        max_completion_tokens=1500,
                        temperature=0.6,
                        stream=True,
                    )
                    full_response, ph = "", st.empty()
                    for chunk in stream:
                        delta = chunk.choices[0].delta.content or ""
                        full_response += delta
                        ph.markdown(full_response + "▌")
                    ph.markdown(full_response)

                    r_tokens = count_tokens(full_response)
                    record_tokens(p_tokens, r_tokens)
                    latency  = (time.time() - t0) * 1000

                    badges = '<div class="badge-row">'
                    if search_results:
                        badges += f'<span class="badge b-blue">🔍 {len(search_results)} docs found</span>'
                    badges += f'<span class="badge b-orange">⚡ {selected_model.split("/")[-1][:20]}</span>'
                    badges += f'<span class="badge b-green">⏱ {latency/1000:.1f}s · {r_tokens} tok</span>'
                    badges += '</div>'
                    st.markdown(badges, unsafe_allow_html=True)

                    if search_results:
                        with st.expander("📎 Sources", expanded=False):
                            for r in search_results:
                                st.markdown(f"- [{r['title']}]({r['url']}) `score:{r['score']}`")

                    messages.append({"role": "assistant", "content": full_response})
                    st.session_state.agent_messages = messages

                except groq_errors.AuthenticationError:
                    st.error("❌ Invalid Groq API Key.")
                except groq_errors.RateLimitError:
                    st.error("⏳ Groq rate limit hit — wait and retry.")
                except Exception as e:
                    _log("error", f"Agent error: {e}")
                    st.error(f"⚠️ Error: {e}")
