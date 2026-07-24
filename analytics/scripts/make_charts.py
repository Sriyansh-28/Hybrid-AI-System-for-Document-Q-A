import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPORT_DIR = ROOT / "exports"
CHART_DIR = ROOT / "charts"

NAVY = "#1f3a5f"
TEAL = "#2a9d8f"
CORAL = "#e76f51"
SAND = "#e9c46a"


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)


def monthly_revenue():
    df = pd.read_csv(EXPORT_DIR / "01_monthly_revenue_growth.csv")
    df = df[df["orders"] >= 100].copy()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(df["month"], df["revenue"] / 1000, marker="o", color=NAVY, linewidth=2)
    ax.fill_between(df["month"], df["revenue"] / 1000, color=NAVY, alpha=0.08)
    style_axes(ax)
    ax.set_title("Monthly Revenue (BRL thousands)", fontsize=13, weight="bold")
    ax.set_ylabel("Revenue (R$ thousands)")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "monthly_revenue.png", dpi=120)
    plt.close(fig)


def top_categories():
    df = pd.read_csv(EXPORT_DIR / "02_top_categories_by_revenue.csv").head(10)
    df = df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(df["category"], df["product_revenue"] / 1000, color=TEAL)
    style_axes(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color="#dddddd", linewidth=0.8)
    ax.set_title("Top 10 Categories by Product Revenue", fontsize=13, weight="bold")
    ax.set_xlabel("Revenue (R$ thousands)")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "top_categories.png", dpi=120)
    plt.close(fig)


def delivery_by_state():
    df = pd.read_csv(EXPORT_DIR / "04_delivery_performance_by_state.csv")
    colors = [CORAL if v >= 10 else NAVY for v in df["late_delivery_pct"]]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(df["state"], df["late_delivery_pct"], color=colors)
    style_axes(ax)
    ax.axhline(df["late_delivery_pct"].mean(), color="#888888", linestyle="--", linewidth=1)
    ax.set_title("Late-Delivery Rate by Customer State (%)", fontsize=13, weight="bold")
    ax.set_ylabel("Late deliveries (%)")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "late_delivery_by_state.png", dpi=120)
    plt.close(fig)


def review_vs_delivery():
    df = pd.read_csv(EXPORT_DIR / "06_review_score_vs_delivery.csv")
    labels = df["delivery_bucket"]
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    width = 0.38
    ax.bar([i - width / 2 for i in x], df["avg_review_score"], width, label="Avg review score", color=NAVY)
    ax.bar([i + width / 2 for i in x], df["pct_negative_score"] / 20, width, label="% negative (scaled)", color=CORAL)
    style_axes(ax)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_title("Review Score by Delivery Timing", fontsize=13, weight="bold")
    for i, row in df.iterrows():
        ax.text(i - width / 2, row["avg_review_score"] + 0.05, f"{row['avg_review_score']:.2f}", ha="center", fontsize=9)
        ax.text(i + width / 2, row["pct_negative_score"] / 20 + 0.05, f"{row['pct_negative_score']:.0f}%", ha="center", fontsize=9)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "review_vs_delivery.png", dpi=120)
    plt.close(fig)


def data_quality():
    dq = pd.read_csv(EXPORT_DIR / "q4_referential_integrity.csv")
    nulls = pd.read_csv(EXPORT_DIR / "q2_null_patterns.csv")
    dups = pd.read_csv(EXPORT_DIR / "q1_duplicate_records.csv")
    items = [
        ("Orders without items", int(dq.loc[dq["integrity_check"] == "orders_without_items", "violations"].iloc[0])),
        ("Orders missing delivery date", int(nulls.loc[nulls["column_checked"] == "orders.order_delivered_customer_date", "null_count"].iloc[0])),
        ("Products missing category", int(nulls.loc[nulls["column_checked"] == "products.category_name", "null_count"].iloc[0])),
        ("Duplicate review_id rows", int(dups.loc[dups["issue"] == "duplicate_review_id", "affected_rows"].iloc[0])),
        ("Orders with multiple reviews", int(dups.loc[dups["issue"] == "orders_with_multiple_reviews", "affected_rows"].iloc[0])),
    ]
    items.sort(key=lambda t: t[1])
    labels = [i[0] for i in items]
    values = [i[1] for i in items]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(labels, values, color=SAND, edgecolor=NAVY)
    style_axes(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color="#dddddd", linewidth=0.8)
    ax.set_title("Data-Quality Issues Detected (row counts)", fontsize=13, weight="bold")
    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2, f"{v:,}", va="center", fontsize=9)
    ax.margins(x=0.15)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "data_quality.png", dpi=120)
    plt.close(fig)


def main():
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    monthly_revenue()
    top_categories()
    delivery_by_state()
    review_vs_delivery()
    data_quality()
    print(f"charts written -> {CHART_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
