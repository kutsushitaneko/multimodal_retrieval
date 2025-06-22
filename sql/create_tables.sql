-- IMAGES テーブル
CREATE TABLE IMAGES (
    image_id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    file_name VARCHAR2(255),
    caption VARCHAR2(4000),
    caption_embedding VECTOR,
    image_data BLOB,
    image_embedding VECTOR,
    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT image_data_not_null CHECK (image_data IS NOT NULL)
);

-- インデックスの作成
CREATE INDEX idx_images_file_name ON IMAGES(file_name);
CREATE INDEX idx_images_upload_date ON IMAGES(upload_date);

-- Vector indexの作成
CREATE VECTOR INDEX idx_image_embedding
ON IMAGES (image_embedding)
ORGANIZATION NEIGHBOR PARTITIONS
WITH DISTANCE DOT
WITH TARGET ACCURACY 95;

CREATE VECTOR INDEX idx_caption_embedding
ON IMAGES (caption_embedding)
ORGANIZATION NEIGHBOR PARTITIONS
WITH DISTANCE DOT
WITH TARGET ACCURACY 95;

-- 全文検索インデックスの作成
begin
    ctx_ddl.create_preference('lexerpref4japanese','JAPANESE_VGRAM_LEXER');
    ctx_ddl.set_attribute('lexerpref4japanese','BIGRAM','TRUE');
end;
/
CREATE INDEX idx_image_caption
ON IMAGES(caption) 
INDEXTYPE IS CTXSYS.CONTEXT
PARAMETERS ('LEXER lexerpref4japanese SYNC (ON COMMIT) DATASTORE CTXSYS.DIRECT_DATASTORE');