SELECT 'duplicate_review_id' AS issue,
       COUNT(*) AS affected_groups,
       SUM(cnt) AS affected_rows
FROM (SELECT review_id, COUNT(*) AS cnt
      FROM order_reviews
      GROUP BY review_id
      HAVING COUNT(*) > 1)
UNION ALL
SELECT 'orders_with_multiple_reviews',
       COUNT(*),
       SUM(cnt)
FROM (SELECT order_id, COUNT(*) AS cnt
      FROM order_reviews
      GROUP BY order_id
      HAVING COUNT(*) > 1)
UNION ALL
SELECT 'fully_duplicate_review_rows',
       COUNT(*),
       SUM(cnt)
FROM (SELECT review_id, order_id, review_score, COUNT(*) AS cnt
      FROM order_reviews
      GROUP BY review_id, order_id, review_score
      HAVING COUNT(*) > 1);
