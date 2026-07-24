# Olist E-Commerce — Business Findings

Analysis of 99,441 orders placed on the Olist Brazilian marketplace between
September 2016 and October 2018. Every number below is produced by a query in
`analysis/` and exported to `exports/`. The query behind each finding is named
in *Source*.

**Headline numbers** (`analysis/00_business_overview.sql`):
gross merchandise value **R$15.7M**, **99,441** orders, **96,096** unique
customers, **8.1%** of delivered orders arrived late.

---

## Finding 1 — Revenue grew ~20x in the first year, then flattened in 2018

Monthly revenue climbed from about **R$137k** in January 2017 to a peak of
**R$1.17M** in November 2017 (Black Friday). Through 2018 it held between
**R$1.0M and R$1.16M** every month with no further growth — April through
August 2018 were essentially flat (month-over-month change stayed within
±11%).

**What it means:** the business finished its land-grab phase. The engine that
drove 2017 growth stopped adding new revenue in 2018.

**Recommendation:** treat 2018 as a plateau, not a seasonal dip. Shift budget
from raw acquisition toward repeat purchase and basket size, because more
first orders are no longer moving the top line.

*Source: `analysis/01_monthly_revenue_growth.sql` → `exports/01_monthly_revenue_growth.csv`*

---

## Finding 2 — Revenue is concentrated in a handful of categories

The **top 10 of 70+ product categories generate 62.4%** of all product
revenue. Health & beauty (**R$1.26M**), watches & gifts (**R$1.21M**) and
bed/bath/table (**R$1.04M**) lead. Watches & gifts earns the most per item
(**R$201 average**) while bed/bath/table sells the most units (11,115) at a
low **R$93 average**.

**What it means:** a small set of categories carries the marketplace. High-
ticket categories and high-volume categories are different businesses that
need different playbooks.

**Recommendation:** protect the top 10 with dedicated inventory and seller
support. Push high-margin categories like watches & gifts in marketing, and
use high-volume categories like bed/bath/table to drive repeat visits.

*Source: `analysis/02_top_categories_by_revenue.sql` → `exports/02_top_categories_by_revenue.csv`*

---

## Finding 3 — Late delivery is a regional problem, not a national one

Nationally **8.1%** of delivered orders arrive after the promised date, but
the rate ranges from **5.0% in Paraná** to **19.7% in Maranhão**. The worst
five states — Maranhão, Ceará, Bahia, Rio de Janeiro and Pará — are all in the
north/northeast and each runs **12–20%** late. Rio de Janeiro is the standout
risk: a large market (12,350 delivered orders) running **13.5%** late.

**What it means:** the promise-date model is miscalibrated for northern
regions and for Rio, where volume makes the problem expensive.

**Recommendation:** lengthen quoted delivery windows for the northern states
so estimates are honest, and audit the Rio de Janeiro carrier lane first — it
is the single largest pool of at-risk orders.

*Source: `analysis/04_delivery_performance_by_state.sql` → `exports/04_delivery_performance_by_state.csv`*

---

## Finding 4 — Late delivery is the strongest driver of bad reviews

Orders delivered on time average **4.30 out of 5** and only **9.2%** score
1–2 stars. Orders that arrive late average **2.57** and **53.9%** score 1–2
stars — a **1.7-star** drop and roughly **6x** more negative reviews.

**What it means:** delivery speed, not product quality, is what customers
punish. A late order is more likely than not to become a negative review.

**Recommendation:** make on-time delivery the primary customer-satisfaction
metric. Fixing the regional delays in Finding 3 is also the highest-leverage
way to lift review scores.

*Source: `analysis/06_review_score_vs_delivery.sql` → `exports/06_review_score_vs_delivery.csv`*

---

## Finding 5 — Almost all customers buy exactly once

**97.0%** of customers placed a single order. Repeat customers are just
**3.0%** of the base and contribute only **5.7%** of revenue. The marketplace
runs almost entirely on one-time buyers.

**What it means:** there is effectively no retention engine. Every month of
revenue must be re-earned from new customers — which is exactly why 2018 went
flat once acquisition slowed (Finding 1).

**Recommendation:** stand up basic lifecycle marketing — post-purchase
follow-up, category-based reorder prompts, a loyalty incentive. Moving repeat
buyers from 3% to even 6% would meaningfully change the revenue trajectory.

*Source: `analysis/07_repeat_vs_onetime_customers.sql` → `exports/07_repeat_vs_onetime_customers.csv`*

---

## Finding 6 — Payments run on credit cards and installments

Credit cards handle **73.9%** of payments and **78.3%** of value, averaging
**3.5 installments** per order. Boleto (a Brazilian bank slip) is the only
other major method at **19.0%** of payments. Debit cards and vouchers together
are under **8%**.

**What it means:** customers rely on installment credit to afford purchases,
so checkout economics are tied to card acceptance and installment terms.

**Recommendation:** keep installment credit friction-free at checkout and
negotiate card processing fees as a priority cost line. Boleto matters enough
(one in five payments) that any boleto outage directly dents conversion.

*Source: `analysis/05_payment_type_mix.sql` → `exports/05_payment_type_mix.csv`*

---

## Finding 7 — Seller leadership is spread thin across states

The strongest seller in each state was ranked separately. São Paulo's leader
handles **1,132 orders (R$250k)**, but in **20 of 23 states the top seller
does fewer than 500 orders**, and the median state-leader manages just **67**.
Outside São Paulo and its neighbours, no single seller reaches national scale.

**What it means:** supply depth is uneven. Many states depend on a thin bench
of sellers, which is a fragility risk if a top seller leaves.

**Recommendation:** recruit and onboard sellers in under-served states, and
give each state's leading seller retention attention — losing one would leave
a visible gap in local supply.

*Source: `analysis/03_top_seller_per_state.sql` → `exports/03_top_seller_per_state.csv`*

---

## Finding 8 — Category momentum is cooling in 2018, matching the plateau

Quarter-over-quarter, the top categories grew fast through 2017 and then
decelerated in 2018. By the second quarter of 2018 the growth column had
already turned negative for computers/accessories (**-42.3%**) and
sports/leisure (**-30.8%**), while the rest slowed to single or low-double
digit gains. (The 2018-Q3 figures fall further because the dataset ends in
mid-October, so that quarter is only partially recorded — read the Q1→Q2
trend, not the truncated final quarter.)

**What it means:** the 2018 plateau in Finding 1 is not caused by one weak
category — the deceleration is broad, across the categories that matter most.

**Recommendation:** do not wait for a single category to rebound. The growth
levers are cross-cutting: retention (Finding 5) and delivery reliability
(Findings 3–4), not a category-specific fix.

*Source: `analysis/08_category_momentum_qoq.sql` → `exports/08_category_momentum_qoq.csv`*

---

## Data-quality findings

Quantified checks run before trusting any of the numbers above.
*Sources: `analysis/quality/q1`–`q4` → `exports/q1`–`q4`.*

- **Duplicates.** **789** `review_id` values repeat, covering **1,603** rows;
  **547** orders carry more than one review (1,098 rows). No fully-identical
  duplicate rows exist, so these are genuine multi-review orders, not import
  errors — the analysis in Finding 4 collapses each order to one score to
  avoid double counting. *(`q1_duplicate_records`)*

- **Null patterns.** **2,965** orders (**2.98%**) have no customer-delivery
  date, almost all still in transit or canceled. A sharper anomaly: **8**
  orders are marked `delivered` yet have **no delivery date** — a status
  contradiction. **610** products (**1.85%**) have no category. *(`q2_null_patterns`)*

- **Outliers.** Beyond mean + 3 standard deviations there are **1,966**
  item prices (max **R$6,735** vs a **R$121** average), **2,041** freight
  charges, and **961** product weights (max **40.4 kg**). These are plausible
  high-end items, not corruption, but they must be capped or segmented before
  averaging. *(`q3_outlier_values`)*

- **Referential integrity.** **775** orders have no line items and **1** order
  has no payment record — these cannot be revenue-analysed and are excluded
  from revenue queries. **3** payments have type `not_defined` (all R$0). All
  order-item product and seller references resolve cleanly (**0** breaks), so
  the join graph is sound. *(`q4_referential_integrity`)*
