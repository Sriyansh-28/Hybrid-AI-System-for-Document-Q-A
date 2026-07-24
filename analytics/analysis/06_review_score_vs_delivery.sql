WITH order_review AS (
    SELECT o.order_id,
           MAX(r.review_score) AS review_score,
           CASE WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date
                THEN 'late' ELSE 'on_time' END AS delivery_bucket
    FROM orders o
    JOIN order_reviews r ON r.order_id = o.order_id
    WHERE o.order_status = 'delivered'
      AND o.order_delivered_customer_date IS NOT NULL
    GROUP BY o.order_id, delivery_bucket
)
SELECT delivery_bucket,
       COUNT(*) AS reviewed_orders,
       ROUND(AVG(review_score), 2) AS avg_review_score,
       ROUND(100.0 * SUM(CASE WHEN review_score <= 2 THEN 1 ELSE 0 END)
             / COUNT(*), 1) AS pct_negative_score,
       ROUND(100.0 * SUM(CASE WHEN review_score = 5 THEN 1 ELSE 0 END)
             / COUNT(*), 1) AS pct_five_star
FROM order_review
GROUP BY delivery_bucket
ORDER BY delivery_bucket;
