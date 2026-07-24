WITH category_sales AS (
    SELECT COALESCE(t.category_name_english, p.category_name, 'unknown') AS category,
           COUNT(*) AS items_sold,
           COUNT(DISTINCT i.order_id) AS orders,
           ROUND(SUM(i.price), 2) AS product_revenue,
           ROUND(AVG(i.price), 2) AS avg_item_price
    FROM order_items i
    JOIN products p ON p.product_id = i.product_id
    LEFT JOIN category_translation t ON t.category_name = p.category_name
    GROUP BY category
)
SELECT RANK() OVER (ORDER BY product_revenue DESC) AS revenue_rank,
       category,
       items_sold,
       orders,
       product_revenue,
       avg_item_price
FROM category_sales
ORDER BY product_revenue DESC
LIMIT 15;
