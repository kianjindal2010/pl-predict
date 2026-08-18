"""Player-data scrape job: Understat per-match (all seasons) + Transfermarkt."""

import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from pl_predict.pipeline.scrape import (
    scrape_transfermarkt,
    scrape_understat_player_matches,
)

print("=== 1/2 UNDERSTAT PER-MATCH (resume) ===", flush=True)

t0 = time.time()
pf, sf = scrape_understat_player_matches()
print(
    "understat done in %.1fs -> players %d, shots %d"
    % (time.time() - t0, len(pf), len(sf)),
    flush=True,
)

print("\n=== 2/2 TRANSFERMARKT SQUAD VALUES ===", flush=True)
t0 = time.time()
df = scrape_transfermarkt(delay=1.0)
print(
    "transfermarkt done in %.1fs -> %d rows" % (time.time() - t0, len(df)), flush=True
)
filled = df.filter(__import__("polars").col("squad_value").is_not_null()).height
print("squad_value non-null: %d/%d" % (filled, df.height), flush=True)
print("ALL SCRAPING DONE", flush=True)
