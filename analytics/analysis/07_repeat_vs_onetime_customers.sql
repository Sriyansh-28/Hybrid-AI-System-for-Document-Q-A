WITH customer_orders AS (
    SELECT c.customer_unique_id AS customer,
           COUNT(DISTINCT o.order_id) AS order_count,
           SUM(i.price + i.freight_value) AS revenue
    FROM orders o
    JOIN customers c ON c.customer_id = o.customer_id
    JOIN order_items i ON i.order_id = o.order_id
    WHERE o.order_status NOT IN ('canceled', 'unavailable')
    GROUP BY c.customer_unique_id
    HAVING SUM(i.price + i.freight_value) > 0
),
segmented AS (
    SELECT CASE WHEN order_count = 1 THEN 'one_time' ELSE 'repeat' END AS segment,
           order_count,
           revenue
    FROM customer_orders
)
SELECT segment,
       COUNT(*) AS customers,
       ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM segmented), 1) AS pct_customers,
       SUM(order_count) AS total_orders,
       ROUND(SUM(revenue), 2) AS total_revenue,
       ROUND(100.0 * SUM(revenue)
             / (SELECT SUM(revenue) FROM segmented), 1) AS pct_revenue
FROM segmented
GROUP BY segment
ORDER BY segment;
