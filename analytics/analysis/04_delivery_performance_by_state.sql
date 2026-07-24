SELECT c.customer_state AS state,
       COUNT(*) AS delivered_orders,
       ROUND(AVG(julianday(o.order_delivered_customer_date)
                 - julianday(o.order_purchase_timestamp)), 1) AS avg_delivery_days,
       ROUND(AVG(julianday(o.order_estimated_delivery_date)
                 - julianday(o.order_delivered_customer_date)), 1) AS avg_days_vs_estimate,
       ROUND(100.0 * SUM(CASE WHEN o.order_delivered_customer_date
                                   > o.order_estimated_delivery_date
                              THEN 1 ELSE 0 END) / COUNT(*), 1) AS late_delivery_pct
FROM orders o
JOIN customers c ON c.customer_id = o.customer_id
WHERE o.order_status = 'delivered'
  AND o.order_delivered_customer_date IS NOT NULL
GROUP BY c.customer_state
HAVING COUNT(*) >= 500
ORDER BY late_delivery_pct DESC;
