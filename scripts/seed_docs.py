"""Write sample documents into data/raw_docs so the FAISS pipeline can be demoed.

Run: python -m scripts.seed_docs
"""


from app.config import get_settings

SAMPLES = {
    "company_background.md": """# Company Background

Acme Analytics is a data-intelligence firm founded in 2018 with 120 employees.
Headquartered in Bengaluru, it serves 340 enterprise clients across retail,
logistics, and fintech. Annual recurring revenue reached $24M in FY2025,
growing 34% year over year. Gross margin is 71%. The company plans to expand
into the EU market in the next two fiscal years, targeting a 20% revenue share
from international operations by 2028.
""",
    "market_trends.md": """# Market Trends in Data Intelligence

The global data intelligence market is projected to grow from $18B in 2025 to
$38B by 2030 (CAGR 16%). Demand is driven by generative-AI analytics, real-time
decision infrastructure, and stricter data-governance regulation (EU AI Act).
Competitors include DataCorp, InsightFlow, and VectorLake. Pricing pressure is
highest in the mid-market segment; the enterprise segment favors integrated
platforms over point tools.
""",
    "product_metrics.md": """# Product Metrics Dashboard Notes

The decision-intelligence product shows 94% weekly active retention and an
average commissioning time of 11 minutes. P95 latency for queries is 1.8s.
Feature flags experiments reveal that the 'auto-insight' surface lifts
user-reported value scores by 12 points. NPS stands at 61.
""",
}


def main() -> int:
    settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    for name, content in SAMPLES.items():
        (settings.data_dir / name).write_text(content.lstrip(), encoding="utf-8")
        print(f"wrote {settings.data_dir / name}")
    print("\nNow build the index with: python -m scripts.index_docs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
