import argparse
from io import StringIO
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from ml.paths import SOURCE_DATA_DIR


def _read_html_table(table) -> pd.DataFrame:
    return pd.read_html(StringIO(str(table)))[0]


def _soup_for(path: Path) -> BeautifulSoup:
    return BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")


def _remove_repeated_headers(soup: BeautifulSoup) -> None:
    for row in soup.find_all("tr", class_=["over_header", "thead"]):
        row.decompose()


def parse_mvp_year(path: Path, year: int) -> pd.DataFrame | None:
    soup = _soup_for(path)
    _remove_repeated_headers(soup)
    table = soup.find(id="mvp")
    if table is None:
        print(f"No MVP table found for {year}; treating as no voting data yet.")
        return None
    frame = _read_html_table(table)
    frame["Year"] = year
    return frame


def parse_player_year(path: Path, year: int) -> pd.DataFrame | None:
    soup = _soup_for(path)
    _remove_repeated_headers(soup)
    table = soup.find(id="per_game_stats")
    if table is None:
        print(f"No player table found for {year}.")
        return None
    frame = _read_html_table(table)
    if "Tm" not in frame.columns and "Team" in frame.columns:
        frame = frame.rename(columns={"Team": "Tm"})
    frame = frame[frame["Rk"].astype(str) != "Rk"].copy()
    frame = frame[frame["Player"].notna()].copy()
    frame = frame[frame["Player"] != "League Average"].copy()
    frame["Year"] = year
    return frame


def parse_advanced_year(path: Path, year: int) -> pd.DataFrame | None:
    soup = _soup_for(path)
    _remove_repeated_headers(soup)
    table = soup.find(id="advanced")
    if table is None:
        print(f"No advanced table found for {year}.")
        return None
    frame = _read_html_table(table)
    if "Tm" not in frame.columns and "Team" in frame.columns:
        frame = frame.rename(columns={"Team": "Tm"})
    frame = frame[frame["Rk"].astype(str) != "Rk"].copy()
    frame = frame[frame["Player"].notna()].copy()
    frame = frame[frame["Player"] != "League Average"].copy()
    frame["Year"] = year
    return frame


def parse_team_year(path: Path, year: int) -> list[pd.DataFrame]:
    frames = []
    for table_id, conference_col in [
        ("divs_standings_E", "Eastern Conference"),
        ("divs_standings_W", "Western Conference"),
    ]:
        soup = _soup_for(path)
        _remove_repeated_headers(soup)
        table = soup.find(id=table_id)
        if table is None:
            print(f"No {table_id} standings table found for {year}.")
            continue
        frame = _read_html_table(table)
        frame["Year"] = year
        frame["Team"] = frame[conference_col]
        frame = frame.drop(columns=[conference_col])
        frames.append(frame)
    return frames


def parse_raw_html(
    start_year: int, end_year: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mvps = []
    players = []
    advanced = []
    teams = []

    for year in range(start_year, end_year + 1):
        mvp_path = SOURCE_DATA_DIR / "mvp" / f"{year}.html"
        player_path = SOURCE_DATA_DIR / "player" / f"{year}.html"
        advanced_path = SOURCE_DATA_DIR / "advanced" / f"{year}.html"
        team_path = SOURCE_DATA_DIR / "team" / f"{year}.html"

        if mvp_path.exists():
            mvp = parse_mvp_year(mvp_path, year)
            if mvp is not None:
                mvps.append(mvp)
        if player_path.exists():
            player = parse_player_year(player_path, year)
            if player is not None:
                players.append(player)
        if advanced_path.exists():
            advanced_year = parse_advanced_year(advanced_path, year)
            if advanced_year is not None:
                advanced.append(advanced_year)
        if team_path.exists():
            teams.extend(parse_team_year(team_path, year))

    if not players:
        raise ValueError("No player data parsed. Run `python -m ml.scrape` first.")
    if not teams:
        raise ValueError("No team data parsed. Run `python -m ml.scrape` first.")

    mvp_frame = pd.concat(mvps, ignore_index=True) if mvps else pd.DataFrame()
    player_frame = pd.concat(players, ignore_index=True)
    advanced_frame = pd.concat(advanced, ignore_index=True) if advanced else pd.DataFrame()
    team_frame = pd.concat(teams, ignore_index=True)
    return mvp_frame, player_frame, advanced_frame, team_frame


def single_team(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.shape[0] == 1:
        return frame
    total = frame[frame["Tm"] == "TOT"].copy()
    if total.empty:
        return frame.tail(1)
    total.loc[:, "Tm"] = frame.iloc[-1]["Tm"]
    return total


def load_abbreviations() -> dict[str, str]:
    abbreviations = pd.read_csv(SOURCE_DATA_DIR / "abbreviations.csv")
    return dict(zip(abbreviations["Abbreviation"], abbreviations["Name"]))


def build_player_mvp_stats(
    mvps: pd.DataFrame,
    players: pd.DataFrame,
    teams: pd.DataFrame,
    advanced: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if mvps.empty:
        mvps = pd.DataFrame(columns=["Player", "Year", "Pts Won", "Pts Max", "Share"])
    else:
        mvps = mvps[["Player", "Year", "Pts Won", "Pts Max", "Share"]].copy()

    players = players.copy()
    players = players.drop(
        columns=[col for col in ["Unnamed: 0", "Rk", "Awards"] if col in players.columns]
    )
    players["Player"] = players["Player"].str.replace("*", "", regex=False)
    players = pd.concat(
        [single_team(group) for _, group in players.groupby(["Player", "Year"], sort=False)],
        ignore_index=True,
    )

    if advanced is not None and not advanced.empty:
        advanced = advanced.copy()
        advanced = advanced.drop(
            columns=[
                col
                for col in ["Unnamed: 0", "Rk", "Awards"]
                if col in advanced.columns or col.startswith("Unnamed:")
            ],
            errors="ignore",
        )
        advanced["Player"] = advanced["Player"].str.replace("*", "", regex=False)
        advanced = pd.concat(
            [
                single_team(group)
                for _, group in advanced.groupby(["Player", "Year"], sort=False)
            ],
            ignore_index=True,
        )
        advanced_keep = [
            col
            for col in [
                "Player",
                "Year",
                "Tm",
                "PER",
                "TS%",
                "3PAr",
                "FTr",
                "ORB%",
                "DRB%",
                "TRB%",
                "AST%",
                "STL%",
                "BLK%",
                "TOV%",
                "USG%",
                "OWS",
                "DWS",
                "WS",
                "WS/48",
                "OBPM",
                "DBPM",
                "BPM",
                "VORP",
            ]
            if col in advanced.columns
        ]
        players = players.merge(
            advanced[advanced_keep],
            how="left",
            on=["Player", "Year", "Tm"],
        )

    combined = players.merge(mvps, how="left", on=["Player", "Year"])
    combined[["Pts Won", "Pts Max", "Share"]] = combined[
        ["Pts Won", "Pts Max", "Share"]
    ].fillna(0)

    teams = teams.copy()
    teams = teams[~teams["W"].astype(str).str.contains("Division", na=False)]
    teams["Team"] = (
        teams["Team"]
        .astype(str)
        .str.replace("\xa0", " ", regex=False)
        .str.replace("*", "", regex=False)
        .str.replace(r"\s*\(\d+\)", "", regex=True)
        .str.strip()
    )

    combined["Team"] = combined["Tm"].map(load_abbreviations())
    stats = combined.merge(teams, how="left", on=["Team", "Year"])
    for column in stats.columns:
        if column not in {"Player", "Pos", "Tm", "Team"}:
            stats[column] = pd.to_numeric(stats[column], errors="coerce")
    stats["GB"] = (
        stats["GB"]
        .astype(str)
        .str.replace("â€”", "0.0", regex=False)
        .str.replace("—", "0.0", regex=False)
    )
    stats["GB"] = pd.to_numeric(stats["GB"], errors="coerce").fillna(0)
    return stats


def rebuild_csvs(start_year: int = 2003, end_year: int = 2026) -> pd.DataFrame:
    mvps, players, advanced, teams = parse_raw_html(start_year, end_year)
    mvps.to_csv(SOURCE_DATA_DIR / "mvps.csv", index=False)
    players.to_csv(SOURCE_DATA_DIR / "players.csv", index=False)
    if not advanced.empty:
        advanced.to_csv(SOURCE_DATA_DIR / "advanced.csv", index=False)
    teams.to_csv(SOURCE_DATA_DIR / "teams.csv", index=False)

    stats = build_player_mvp_stats(mvps, players, teams, advanced=advanced)
    stats.to_csv(SOURCE_DATA_DIR / "player_mvp_stats.csv", index=False)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse cached Basketball Reference HTML into CSVs.")
    parser.add_argument("--start-year", type=int, default=2003)
    parser.add_argument("--end-year", type=int, default=2026)
    args = parser.parse_args()

    stats = rebuild_csvs(start_year=args.start_year, end_year=args.end_year)
    print(
        f"Saved player_mvp_stats.csv with {stats.shape[0]} rows, "
        f"{stats.shape[1]} columns, years {int(stats['Year'].min())}-{int(stats['Year'].max())}."
    )


if __name__ == "__main__":
    main()
