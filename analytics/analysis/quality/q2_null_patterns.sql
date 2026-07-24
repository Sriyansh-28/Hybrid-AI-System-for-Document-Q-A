SELECT 'orders.order_approved_at' AS column_checked,
       SUM(CASE WHEN order_approved_at IS NULL THEN 1 ELSE 0 END) AS null_count,
       COUNT(*) AS total_rows,
       ROUND(100.0 * SUM(CASE WHEN order_approved_at IS NULL THEN 1 ELSE 0 END)
             / COUNT(*), 2) AS null_pct
FROM orders
UNION ALL
SELECT 'orders.order_delivered_customer_date',
       SUM(CASE WHEN order_delivered_customer_date IS NULL THEN 1 ELSE 0 END),
       COUNT(*),
       ROUND(100.0 * SUM(CASE WHEN order_delivered_customer_date IS NULL THEN 1 ELSE 0 END)
             / COUNT(*), 2)
FROM orders
UNION ALL
SELECT 'orders.delivered_status_missing_date',
       COUNT(*),
       (SELECT COUNT(*) FROM orders WHERE order_status = 'delivered'),
       ROUND(100.0 * COUNT(*)
             / (SELECT COUNT(*) FROM orders WHERE order_status = 'delivered'), 2)
FROM orders
WHERE order_status = 'delivered'
  AND order_delivered_customer_date IS NULL
UNION ALL
SELECT 'products.category_name',
       SUM(CASE WHEN category_name IS NULL THEN 1 ELSE 0 END),
       COUNT(*),
       ROUND(100.0 * SUM(CASE WHEN category_name IS NULL THEN 1 ELSE 0 END)
             / COUNT(*), 2)
FROM products
UNION ALL
SELECT 'products.product_weight_g',
       SUM(CASE WHEN product_weight_g IS NULL THEN 1 ELSE 0 END),
       COUNT(*),
       ROUND(100.0 * SUM(CASE WHEN product_weight_g IS NULL THEN 1 ELSE 0 END)
             / COUNT(*), 2)
FROM products
ORDER BY null_pct DESC;
