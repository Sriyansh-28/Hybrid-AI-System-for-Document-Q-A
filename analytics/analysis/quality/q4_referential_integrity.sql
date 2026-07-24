SELECT 'orders_without_items' AS integrity_check,
       COUNT(*) AS violations
FROM orders o
WHERE NOT EXISTS (SELECT 1 FROM order_items i WHERE i.order_id = o.order_id)
UNION ALL
SELECT 'orders_without_payment',
       COUNT(*)
FROM orders o
WHERE NOT EXISTS (SELECT 1 FROM order_payments p WHERE p.order_id = o.order_id)
UNION ALL
SELECT 'items_referencing_missing_product',
       COUNT(*)
FROM order_items i
WHERE NOT EXISTS (SELECT 1 FROM products p WHERE p.product_id = i.product_id)
UNION ALL
SELECT 'items_referencing_missing_seller',
       COUNT(*)
FROM order_items i
WHERE NOT EXISTS (SELECT 1 FROM sellers s WHERE s.seller_id = i.seller_id)
UNION ALL
SELECT 'reviews_referencing_missing_order',
       COUNT(*)
FROM order_reviews r
WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.order_id = r.order_id)
UNION ALL
SELECT 'payments_referencing_missing_order',
       COUNT(*)
FROM order_payments p
WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.order_id = p.order_id)
UNION ALL
SELECT 'payment_type_not_defined',
       COUNT(*)
FROM order_payments
WHERE payment_type = 'not_defined'
ORDER BY violations DESC;
