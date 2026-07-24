SELECT payment_type,
       COUNT(*) AS payment_count,
       ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM order_payments), 1) AS pct_of_payments,
       ROUND(AVG(payment_installments), 2) AS avg_installments,
       ROUND(SUM(payment_value), 2) AS total_value,
       ROUND(100.0 * SUM(payment_value)
             / (SELECT SUM(payment_value) FROM order_payments), 1) AS pct_of_value
FROM order_payments
GROUP BY payment_type
ORDER BY total_value DESC;
