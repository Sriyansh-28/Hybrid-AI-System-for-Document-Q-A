WITH seller_revenue AS (
    SELECT s.seller_id,
           s.seller_state,
           COUNT(DISTINCT i.order_id) AS orders,
           ROUND(SUM(i.price + i.freight_value), 2) AS revenue
    FROM order_items i
    JOIN sellers s ON s.seller_id = i.seller_id
    GROUP BY s.seller_id, s.seller_state
),
ranked AS (
    SELECT seller_state,
           seller_id,
           orders,
           revenue,
           RANK() OVER (PARTITION BY seller_state ORDER BY revenue DESC) AS state_rank
    FROM seller_revenue
)
SELECT seller_state,
       seller_id,
       orders,
       revenue,
       state_rank
FROM ranked
WHERE state_rank = 1
ORDER BY revenue DESC;
