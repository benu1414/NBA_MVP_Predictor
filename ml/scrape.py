import argparse
import time
from pathlib import Path

import requests

from ml.paths import SOURCE_DATA_DIR


URLS = {
    "mvp": "https://www.basketball-reference.com/awards/awards_{year}.html",
    "player": "https://www.basketball-reference.com/leagues/NBA_{year}_per_game.html",
    "advanced": "https://www.basketball-reference.com/leagues/NBA_{year}_advanced.html",
    "team": "https://www.basketball-reference.com/leagues/NBA_{year}_standings.html",
}

HEADERS = {
    "User-Agent": "nba-mvp-predictor/0.2 (+https://github.com/local/nba-mvp-predictor)",
}


def download_page(url: str, path: Path, timeout: int = 30) -> bool:
    response = requests.get(url, headers=HEADERS, timeout=timeout)
    if response.status_code == 404:
        print(f"Skipping missing page: {url}")
        return False
    response.raise_for_status()
    path.write_text(response.content.decode("utf-8", errors="replace"), encoding="utf-8")
    return True


def scrape_years(
    start_year: int,
    end_year: int,
    force: bool = False,
    delay: float = 1.0,
    data_types: list[str] | None = None,
) -> None:
    selected_urls = URLS if data_types is None else {key: URLS[key] for key in data_types}
    for data_type, url_template in selected_urls.items():
        target_dir = SOURCE_DATA_DIR / data_type
        target_dir.mkdir(parents=True, exist_ok=True)

        for year in range(start_year, end_year + 1):
            path = target_dir / f"{year}.html"
            if path.exists() and not force:
                print(f"Already cached {data_type} {year}: {path}")
                continue

            url = url_template.format(year=year)
            try:
                saved = download_page(url, path)
                if saved:
                    print(f"Saved {data_type} {year}: {path}")
            except requests.RequestException as exc:
                print(f"Failed {data_type} {year}: {exc}")

            if delay:
                time.sleep(delay)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Basketball Reference HTML pages.")
    parser.add_argument("--start-year", type=int, default=2003)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--force", action="store_true", help="Redownload pages that already exist.")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds to wait between requests.")
    parser.add_argument(
        "--types",
        nargs="+",
        choices=sorted(URLS),
        help="Limit download to selected data types.",
    )
    args = parser.parse_args()

    scrape_years(
        args.start_year,
        args.end_year,
        force=args.force,
        delay=args.delay,
        data_types=args.types,
    )


if __name__ == "__main__":
    main()
