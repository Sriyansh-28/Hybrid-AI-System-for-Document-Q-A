WITH item_category AS (
    SELECT o.order_purchase_timestamp AS ts,
           i.price + i.freight_value AS revenue,
           COALESCE(t.category_name_english, p.category_name, 'unknown') AS category
    FROM order_items i
    JOIN orders o ON o.order_id = i.order_id
    JOIN products p ON p.product_id = i.product_id
    LEFT JOIN category_translation t ON t.category_name = p.category_name
    WHERE o.order_status NOT IN ('canceled', 'unavailable')
),
top_categories AS (
    SELECT category
    FROM item_category
    GROUP BY category
    ORDER BY SUM(revenue) DESC
    LIMIT 6
),
quarterly AS (
    SELECT category,
           strftime('%Y', ts) || '-Q'
               || ((CAST(strftime('%m', ts) AS INTEGER) + 2) / 3) AS quarter,
           ROUND(SUM(revenue), 2) AS revenue
    FROM item_category
    WHERE category IN (SELECT category FROM top_categories)
    GROUP BY category, quarter
)
SELECT category,
       quarter,
       revenue,
       LAG(revenue) OVER (PARTITION BY category ORDER BY quarter) AS prev_quarter_revenue,
       ROUND(100.0 * (revenue - LAG(revenue) OVER (PARTITION BY category ORDER BY quarter))
             / LAG(revenue) OVER (PARTITION BY category ORDER BY quarter), 1) AS qoq_growth_pct
FROM quarterly
ORDER BY category, quarter;
