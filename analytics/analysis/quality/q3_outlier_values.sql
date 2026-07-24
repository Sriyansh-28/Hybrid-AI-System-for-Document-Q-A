WITH price_stats AS (
    SELECT AVG(price) AS mean_val,
           AVG(price * price) - AVG(price) * AVG(price) AS var_val
    FROM order_items
),
freight_stats AS (
    SELECT AVG(freight_value) AS mean_val,
           AVG(freight_value * freight_value) - AVG(freight_value) * AVG(freight_value) AS var_val
    FROM order_items
),
weight_stats AS (
    SELECT AVG(product_weight_g) AS mean_val,
           AVG(product_weight_g * product_weight_g)
               - AVG(product_weight_g) * AVG(product_weight_g) AS var_val
    FROM products
    WHERE product_weight_g IS NOT NULL
)
SELECT 'order_items.price' AS metric,
       ROUND((SELECT mean_val FROM price_stats), 2) AS mean_value,
       ROUND((SELECT mean_val + 3 * sqrt(var_val) FROM price_stats), 2) AS upper_3sigma,
       COUNT(*) AS outlier_count,
       ROUND(MAX(price), 2) AS max_value
FROM order_items, price_stats
WHERE price > mean_val + 3 * sqrt(var_val)
UNION ALL
SELECT 'order_items.freight_value',
       ROUND((SELECT mean_val FROM freight_stats), 2),
       ROUND((SELECT mean_val + 3 * sqrt(var_val) FROM freight_stats), 2),
       COUNT(*),
       ROUND(MAX(freight_value), 2)
FROM order_items, freight_stats
WHERE freight_value > mean_val + 3 * sqrt(var_val)
UNION ALL
SELECT 'products.product_weight_g',
       ROUND((SELECT mean_val FROM weight_stats), 2),
       ROUND((SELECT mean_val + 3 * sqrt(var_val) FROM weight_stats), 2),
       COUNT(*),
       ROUND(MAX(product_weight_g), 2)
FROM products, weight_stats
WHERE product_weight_g > mean_val + 3 * sqrt(var_val);
