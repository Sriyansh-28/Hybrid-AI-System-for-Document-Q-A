WITH valid_items AS (
    SELECT i.price, i.freight_value, i.product_id
    FROM order_items i
    JOIN orders o ON o.order_id = i.order_id
    WHERE o.order_status NOT IN ('canceled', 'unavailable')
),
delivered AS (
    SELECT order_delivered_customer_date > order_estimated_delivery_date AS is_late
    FROM orders
    WHERE order_status = 'delivered'
      AND order_delivered_customer_date IS NOT NULL
),
cat_rev AS (
    SELECT COALESCE(p.category_name, 'unknown') AS category,
           SUM(i.price) AS rev
    FROM order_items i
    JOIN products p ON p.product_id = i.product_id
    GROUP BY category
)
SELECT 'gross_merchandise_value_brl' AS metric,
       ROUND((SELECT SUM(price + freight_value) FROM valid_items), 2) AS value
UNION ALL
SELECT 'total_orders',
       (SELECT COUNT(*) FROM orders)
UNION ALL
SELECT 'unique_customers',
       (SELECT COUNT(DISTINCT customer_unique_id) FROM customers)
UNION ALL
SELECT 'overall_late_delivery_pct',
       ROUND(100.0 * (SELECT SUM(is_late) FROM delivered)
             / (SELECT COUNT(*) FROM delivered), 1)
UNION ALL
SELECT 'top10_category_revenue_share_pct',
       ROUND(100.0
             * (SELECT SUM(rev) FROM (SELECT rev FROM cat_rev ORDER BY rev DESC LIMIT 10))
             / (SELECT SUM(rev) FROM cat_rev), 1);
