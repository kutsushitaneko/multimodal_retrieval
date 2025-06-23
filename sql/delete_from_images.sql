

-- file_name = 'スライド95.JPG' のレコードの caption を検索
SELECT image_id, file_name, caption, upload_date 
FROM images 
WHERE file_name = 'スライド95.JPG';

-- file_name = 'スライド95.JPG' のレコードの caption を更新
UPDATE images 
SET caption = '新しいキャプション内容'  -- 必要に応じてキャプション内容を変更してください
WHERE file_name = 'スライド95.JPG';
commit;

-- 削除
-- delete from images where file_name = 'スライド95.JPG';
-- commit;