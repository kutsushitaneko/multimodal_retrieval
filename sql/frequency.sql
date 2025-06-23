-- 10. 動的停止語管理（頻出語の自動検出）
WITH frequent_words AS (
  SELECT 
    token,
    COUNT(*) as frequency,
    COUNT(DISTINCT doc_id) as doc_count
  FROM (
    -- この部分は実際のトークン化結果に基づく
    SELECT DISTINCT
      REGEXP_SUBSTR(caption, '\S+', 1, LEVEL) as token,
      image_id as doc_id
    FROM IMAGES
    CONNECT BY REGEXP_SUBSTR(caption, '\S+', 1, LEVEL) IS NOT NULL
    AND PRIOR image_id = image_id 
    AND PRIOR SYS_GUID() IS NOT NULL
  )
  WHERE LENGTH(token) > 1
  GROUP BY token
)
SELECT 
  SUBSTR(token, 1, 30) as token,
  frequency,
  doc_count,
  ROUND(frequency / doc_count, 2) as avg_per_doc
FROM frequent_words
WHERE frequency > 1  -- 頻出閾値
AND doc_count > (SELECT COUNT(*) * 0.1 FROM IMAGES)  -- 80%以上の文書に出現
ORDER BY frequency DESC;