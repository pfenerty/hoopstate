# Bronze schemas

The bronze zone holds **typed tables, 1:1 with their source CSV**. Each dataset
is produced in two steps:

1. `hoopstate.ingest.bulk_loader` downloads the source `tar.xz` and writes a
   **lossless string capture** — every column read as text — to
   `bronze/season=<year>/<name>.parquet`.
2. `hoopstate.ingest.bronze` reads that capture, applies the explicit,
   hand-authored schema below, and atomically rewrites the same path with typed
   columns.

Schemas are hand-authored, not inferred, so the bronze contract is a property of
the code rather than of whichever rows a reader happened to sample. Casts are
strict: a value that does not fit its declared type fails the conversion instead
of becoming a silent null. Every type below was verified against a full-season
(2023-24) scan — the strict cast succeeds over 100% of rows.

Column order matches each source's CSV header exactly. An empty source field is
null in bronze (an empty CSV cell reads as null, not `""`).

## Row counts (2023-24 regular season)

| Dataset          | Rows      | Games | Notes                                        |
| ---------------- | --------- | ----- | -------------------------------------------- |
| `nbastats_2023`  | 567,665   | 1,230 | Full regular season (1,230 games).           |
| `datanba_2023`   | 580,476   | 1,228 | Parallel feed; two games absent, more events per game. |

Row counts are invariant across the string→typed conversion (it is 1:1), so a
typed table's row count is also its source CSV's data-row count.

## `nbastats` — primary event feed (stats.nba.com)

The widest, most descriptive play-by-play, and the backbone of the canonical
model. 34 columns.

| Column | Type | Notes |
| --- | --- | --- |
| `GAME_ID` | Int64 | Integer-encoded here (no leading zeros); join key. |
| `EVENTNUM` | Int64 | Event sequence within a game; join key. |
| `EVENTMSGTYPE` | Int64 | Event category code. |
| `EVENTMSGACTIONTYPE` | Int64 | Event sub-type code. |
| `PERIOD` | Int64 | 1–4 regulation, 5+ overtime. |
| `WCTIMESTRING` | String | Wall-clock time, e.g. `7:11 PM`. |
| `PCTIMESTRING` | String | Game clock, e.g. `12:00`. |
| `HOMEDESCRIPTION` | String | Nullable; blank on non-home events. |
| `NEUTRALDESCRIPTION` | String | Nullable; period start/end markers. |
| `VISITORDESCRIPTION` | String | Nullable; blank on non-visitor events. |
| `SCORE` | String | e.g. `0 - 2`; blank until the first made basket. |
| `SCOREMARGIN` | String | **Kept as text**: signed integers plus the literal `TIE`. |
| `PERSON1TYPE` | Int64 | Role code for player 1. |
| `PLAYER1_ID` | Int64 | `0` means no player (a real sentinel, not null). |
| `PLAYER1_NAME` | String | Nullable. |
| `PLAYER1_TEAM_ID` | Int64 | Nullable (null when player 1 is not a team player). |
| `PLAYER1_TEAM_CITY` | String | Nullable. |
| `PLAYER1_TEAM_NICKNAME` | String | Nullable. |
| `PLAYER1_TEAM_ABBREVIATION` | String | Nullable. |
| `PERSON2TYPE` | Int64 | Role code for player 2. |
| `PLAYER2_ID` | Int64 | `0` means no player. |
| `PLAYER2_NAME` | String | Nullable. |
| `PLAYER2_TEAM_ID` | Int64 | Nullable. |
| `PLAYER2_TEAM_CITY` | String | Nullable. |
| `PLAYER2_TEAM_NICKNAME` | String | Nullable. |
| `PLAYER2_TEAM_ABBREVIATION` | String | Nullable. |
| `PERSON3TYPE` | Int64 | Role code for player 3. |
| `PLAYER3_ID` | Int64 | `0` means no player. |
| `PLAYER3_NAME` | String | Nullable. |
| `PLAYER3_TEAM_ID` | Int64 | Nullable. |
| `PLAYER3_TEAM_CITY` | String | Nullable. |
| `PLAYER3_TEAM_NICKNAME` | String | Nullable. |
| `PLAYER3_TEAM_ABBREVIATION` | String | Nullable. |
| `VIDEO_AVAILABLE_FLAG` | Int64 | `0`/`1`; kept integer to stay 1:1 with source. |

## `datanba` — parallel feed with explicit offense team id

Converted alongside `nbastats` for one column: `oftid`, the **offense team id**
present on every event. It is the independent cross-check that catches the
out-of-order events stats.nba.com is known to contain. 23 columns.

| Column | Type | Notes |
| --- | --- | --- |
| `evt` | Int64 | Event number; joins to `nbastats.EVENTNUM`. |
| `wallclk` | String | ISO-8601, mixed sub-second precision (`...Z`, `....200Z`). Parsing is a silver concern. |
| `cl` | String | Game clock, e.g. `12:00`. |
| `de` | String | Event description. |
| `locX` | Int64 | Court x-coordinate. |
| `locY` | Int64 | Court y-coordinate. |
| `opt1` | Int64 | Source option field. |
| `opt2` | Int64 | Source option field. |
| `opt3` | Int64 | Source option field. |
| `opt4` | Int64 | Source option field. |
| `mtype` | Int64 | Event sub-type code. |
| `etype` | Int64 | Event category code. |
| `opid` | Int64 | Secondary player id; null on most events. |
| `tid` | Int64 | Team id of the acting player. |
| `pid` | Int64 | Acting player id (`0` sentinel). |
| `hs` | Int64 | Home score after the event. |
| `vs` | Int64 | Visitor score after the event. |
| `epid` | Int64 | Tertiary player id; null on most events. |
| `oftid` | Int64 | **Offense team id** — the cross-check against `nbastats`. |
| `ord` | Int64 | Source ordering key. |
| `pts` | Int64 | Points scored on the event (0–3). |
| `PERIOD` | Int64 | 1–4 regulation, 5+ overtime. |
| `GAME_ID` | Int64 | Join key; matches `nbastats.GAME_ID`. |

## Join key

Both feeds cover the same games and join on `(GAME_ID, EVENTNUM/evt)`, which is
why both id columns are typed as integers — the key stays type-consistent across
sources. An inner join on the 2023-24 tables matches 566,129 rows.

Two observations that motivate keeping the second feed, both surfaced by this
join and both left for downstream (E4) work, not bronze:

- `nbastats` carries a small number of duplicate `(GAME_ID, EVENTNUM)` keys
  (3 in 2023-24) — the out-of-order/duplicated events stats.nba.com is known
  for. `datanba` has none.
- `datanba` is missing two games present in `nbastats`, so it is a cross-check,
  not a drop-in replacement.
