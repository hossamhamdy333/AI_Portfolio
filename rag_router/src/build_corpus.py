"""Build the 4-domain corpus from Wikipedia via HuggingFace `datasets`.

Pulls a real dump, filters it down with actual logic (not hand-picked
pages), and saves one clean parquet per domain to data/processed/<domain>/.

Uses the community-maintained `wikimedia/wikipedia` dataset (20231101.en
snapshot) -- properly licensed (CC BY-SA), built for exactly this kind of
downstream ML use, and lets us filter by real category signal instead of
guessing article titles by hand.
"""

import yaml
from datasets import load_dataset
import pandas as pd

# Keyword sets used to match articles into a domain via title/category text.
# Deliberately conservative (specific terms) to keep precision high -- a
# false-positive "sports" article showing up in "tech" would quietly hurt
# the router's routing-accuracy metric later.
DOMAIN_KEYWORDS = {
    "sports": [
        "football", "cricket", "basketball", "tennis", "olympic", "athlete",
        "championship", "tournament", "fifa", "nba", "rugby", "baseball",
    ],
    "tech": [
        "computer", "software", "internet", "artificial intelligence",
        "programming", "semiconductor", "smartphone", "cybersecurity",
        "cloud computing", "database", "algorithm",
    ],
    "history": [
        "war", "empire", "revolution", "dynasty", "ancient", "medieval",
        "kingdom", "civilization", "battle of", "treaty of",
    ],
    "english_literature": [
        "novel", "poet", "poetry", "playwright", "shakespeare", "literary",
        "fiction", "author", "victorian literature", "romantic poetry",
    ],
}

DOMAINS = list(DOMAIN_KEYWORDS.keys())


def article_matches_domain(title, text, keywords):
    haystack = (title + " " + text[:500]).lower()
    return any(kw in haystack for kw in keywords)


def build_domain_corpus(dataset_split, domain_keywords, target_per_domain, seed=42,
                         max_articles_scanned=500_000, progress_every=10_000):
    """Stream the wikipedia dump once, bucket matching articles per domain.

    Streaming (not loading the full ~20GB dump into memory) since we only
    need a few hundred articles per domain out of millions.

    Prints progress every `progress_every` articles scanned, and stops after
    `max_articles_scanned` even if some domains aren't full yet -- without
    this, a domain with rare keyword matches could scan for a very long time
    with zero visible output, indistinguishable from a hang.
    """
    buckets = {domain: [] for domain in domain_keywords}
    filled = set()

    for scanned, row in enumerate(dataset_split, start=1):
        if len(filled) == len(domain_keywords):
            break
        if scanned >= max_articles_scanned:
            print(f"Hit max_articles_scanned={max_articles_scanned}, stopping early.")
            print(f"Bucket sizes so far: {{k: len(v) for k, v in buckets.items()}}")
            break
        if scanned % progress_every == 0:
            sizes = {k: len(v) for k, v in buckets.items()}
            print(f"Scanned {scanned} articles... bucket sizes: {sizes}")

        title, text = row["title"], row["text"]
        for domain, keywords in domain_keywords.items():
            if domain in filled:
                continue
            if article_matches_domain(title, text, keywords):
                buckets[domain].append({"title": title, "text": text, "domain": domain})
                if len(buckets[domain]) >= target_per_domain:
                    filled.add(domain)
                break  # one domain per article, avoid double-counting

    return buckets


def save_domain_parquets(buckets, output_dir):
    for domain, rows in buckets.items():
        df = pd.DataFrame(rows)
        df["article_id"] = [f"{domain}_{i}" for i in range(len(df))]
        out_path = f"{output_dir}/{domain}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"{domain}: {len(df)} articles -> {out_path}")


if __name__ == "__main__":
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    wiki = load_dataset(config["data"]["source_dataset"], config["data"]["source_config"], split="train", streaming=True)
    buckets = build_domain_corpus(
        wiki, DOMAIN_KEYWORDS,
        target_per_domain=config["data"]["target_per_domain"],
        seed=config["data"]["random_seed"],
        max_articles_scanned=config["data"]["max_articles_scanned"],
    )
    save_domain_parquets(buckets, output_dir="data/processed")
