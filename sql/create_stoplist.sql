-- 0. 既存のストップリストを削除（存在する場合）
BEGIN
  BEGIN
    ctx_ddl.drop_stoplist('multilingual_stoplist');
    DBMS_OUTPUT.PUT_LINE('既存のストップリスト multilingual_stoplist を削除しました');
  EXCEPTION
    WHEN OTHERS THEN
      IF SQLCODE = -20000 THEN
        DBMS_OUTPUT.PUT_LINE('ストップリスト multilingual_stoplist は存在しません');
      ELSE
        RAISE;
      END IF;
  END;
END;
/

-- 1. カスタム停止語リストの作成
BEGIN
  -- 基本停止語リストをベースにカスタムリストを作成
  ctx_ddl.create_stoplist('multilingual_stoplist', 'BASIC_STOPLIST');
END;
/

-- 2. 停止語の追加
BEGIN
  -- 助詞・助動詞の追加
  ctx_ddl.add_stopword('multilingual_stoplist', 'は');
  ctx_ddl.add_stopword('multilingual_stoplist', 'が');
  ctx_ddl.add_stopword('multilingual_stoplist', 'を');
  ctx_ddl.add_stopword('multilingual_stoplist', 'に');
  ctx_ddl.add_stopword('multilingual_stoplist', 'で');
  ctx_ddl.add_stopword('multilingual_stoplist', 'と');
  ctx_ddl.add_stopword('multilingual_stoplist', 'から');
  ctx_ddl.add_stopword('multilingual_stoplist', 'まで');
  ctx_ddl.add_stopword('multilingual_stoplist', 'です');
  ctx_ddl.add_stopword('multilingual_stoplist', 'ます');
  ctx_ddl.add_stopword('multilingual_stoplist', 'である');
  ctx_ddl.add_stopword('multilingual_stoplist', 'ですか');
  ctx_ddl.add_stopword('multilingual_stoplist', 'ですよ');
  ctx_ddl.add_stopword('multilingual_stoplist', 'ですね');
  ctx_ddl.add_stopword('multilingual_stoplist', 'でした');
  ctx_ddl.add_stopword('multilingual_stoplist', 'でしたか');
  ctx_ddl.add_stopword('multilingual_stoplist', 'でしたよ');
  ctx_ddl.add_stopword('multilingual_stoplist', 'でしたね');
  ctx_ddl.add_stopword('multilingual_stoplist', 'について');
  ctx_ddl.add_stopword('multilingual_stoplist', 'において');
  ctx_ddl.add_stopword('multilingual_stoplist', 'に関して');
  ctx_ddl.add_stopword('multilingual_stoplist', 'に対して');
  ctx_ddl.add_stopword('multilingual_stoplist', 'だ');
  ctx_ddl.add_stopword('multilingual_stoplist', 'でしょう');
  ctx_ddl.add_stopword('multilingual_stoplist', 'かもしれない');
  
  -- 指示語の追加
  ctx_ddl.add_stopword('multilingual_stoplist', 'これ');
  ctx_ddl.add_stopword('multilingual_stoplist', 'それ');
  ctx_ddl.add_stopword('multilingual_stoplist', 'あれ');
  ctx_ddl.add_stopword('multilingual_stoplist', 'どれ');
  ctx_ddl.add_stopword('multilingual_stoplist', 'ここ');
  ctx_ddl.add_stopword('multilingual_stoplist', 'そこ');
  ctx_ddl.add_stopword('multilingual_stoplist', 'あそこ');
  ctx_ddl.add_stopword('multilingual_stoplist', 'どこ');
  ctx_ddl.add_stopword('multilingual_stoplist', 'こう');
  ctx_ddl.add_stopword('multilingual_stoplist', 'そう');
  ctx_ddl.add_stopword('multilingual_stoplist', 'どう');
  ctx_ddl.add_stopword('multilingual_stoplist', 'あの');
  
  -- 一般的すぎる名詞の追加
  ctx_ddl.add_stopword('multilingual_stoplist', 'もの');
  ctx_ddl.add_stopword('multilingual_stoplist', 'こと');
  ctx_ddl.add_stopword('multilingual_stoplist', 'とき');
  ctx_ddl.add_stopword('multilingual_stoplist', '場合');
  ctx_ddl.add_stopword('multilingual_stoplist', '内容');
  ctx_ddl.add_stopword('multilingual_stoplist', '説明');
  ctx_ddl.add_stopword('multilingual_stoplist', '情報');
  ctx_ddl.add_stopword('multilingual_stoplist', '方法');
  ctx_ddl.add_stopword('multilingual_stoplist', '状況');
  ctx_ddl.add_stopword('multilingual_stoplist', '結果');
  ctx_ddl.add_stopword('multilingual_stoplist', '仕様');
  ctx_ddl.add_stopword('multilingual_stoplist', '使用');
  ctx_ddl.add_stopword('multilingual_stoplist', '使用方法');
  ctx_ddl.add_stopword('multilingual_stoplist', '使用例');
  ctx_ddl.add_stopword('multilingual_stoplist', '使い方');
  ctx_ddl.add_stopword('multilingual_stoplist', '利用');
  ctx_ddl.add_stopword('multilingual_stoplist', '利用方法');
  ctx_ddl.add_stopword('multilingual_stoplist', '利用例');
  ctx_ddl.add_stopword('multilingual_stoplist', '活用');
  ctx_ddl.add_stopword('multilingual_stoplist', '活用方法');
  ctx_ddl.add_stopword('multilingual_stoplist', '活用例');
  ctx_ddl.add_stopword('multilingual_stoplist', '例');
  ctx_ddl.add_stopword('multilingual_stoplist', '例文');
  ctx_ddl.add_stopword('multilingual_stoplist', '例示');
  ctx_ddl.add_stopword('multilingual_stoplist', '機能');
  ctx_ddl.add_stopword('multilingual_stoplist', '詳細');

  -- ドメイン固有の停止後（IT）
  ctx_ddl.add_stopword('multilingual_stoplist', 'データ');
  ctx_ddl.add_stopword('multilingual_stoplist', 'システム');
  ctx_ddl.add_stopword('multilingual_stoplist', 'ソフトウェア');
  ctx_ddl.add_stopword('multilingual_stoplist', 'ハードウェア');
  ctx_ddl.add_stopword('multilingual_stoplist', 'ネットワーク');
  ctx_ddl.add_stopword('multilingual_stoplist', 'セキュリティ');
  ctx_ddl.add_stopword('multilingual_stoplist', 'データベース');
  ctx_ddl.add_stopword('multilingual_stoplist', 'アプリ');
  ctx_ddl.add_stopword('multilingual_stoplist', 'アプリケーション');
  ctx_ddl.add_stopword('multilingual_stoplist', 'ツール');
  ctx_ddl.add_stopword('multilingual_stoplist', 'サービス');
  ctx_ddl.add_stopword('multilingual_stoplist', 'ソリューション');
  ctx_ddl.add_stopword('multilingual_stoplist', 'パラメータ');
  ctx_ddl.add_stopword('multilingual_stoplist', 'パラメーター');
  ctx_ddl.add_stopword('multilingual_stoplist', 'コンピュータ');
  ctx_ddl.add_stopword('multilingual_stoplist', 'コンピューター');
  ctx_ddl.add_stopword('multilingual_stoplist', 'サーバ');
  ctx_ddl.add_stopword('multilingual_stoplist', 'サーバー');
  ctx_ddl.add_stopword('multilingual_stoplist', 'スマホ');
  ctx_ddl.add_stopword('multilingual_stoplist', 'スマートフォン');
  ctx_ddl.add_stopword('multilingual_stoplist', 'モバイル');
  ctx_ddl.add_stopword('multilingual_stoplist', 'モバイルアプリ');
  ctx_ddl.add_stopword('multilingual_stoplist', 'モバイルアプリケーション');
  ctx_ddl.add_stopword('multilingual_stoplist', 'デバイス');
  ctx_ddl.add_stopword('multilingual_stoplist', 'コード');
  ctx_ddl.add_stopword('multilingual_stoplist', 'プログラム');
  ctx_ddl.add_stopword('multilingual_stoplist', 'サンプル');


 -- ドメイン固有の停止後（マークダウンなど）
  ctx_ddl.add_stopword('multilingual_stoplist', '###');
  ctx_ddl.add_stopword('multilingual_stoplist', '**');
  ctx_ddl.add_stopword('multilingual_stoplist', '@');
  ctx_ddl.add_stopword('multilingual_stoplist', '1.');
  ctx_ddl.add_stopword('multilingual_stoplist', '2.');
  ctx_ddl.add_stopword('multilingual_stoplist', '3.');
  ctx_ddl.add_stopword('multilingual_stoplist', '4.');
  ctx_ddl.add_stopword('multilingual_stoplist', '5.');
  ctx_ddl.add_stopword('multilingual_stoplist', '6.');
  ctx_ddl.add_stopword('multilingual_stoplist', '7.');
  ctx_ddl.add_stopword('multilingual_stoplist', '8.');
  ctx_ddl.add_stopword('multilingual_stoplist', '9.');
  ctx_ddl.add_stopword('multilingual_stoplist', '10.');
  ctx_ddl.add_stopword('multilingual_stoplist', '11.');
  ctx_ddl.add_stopword('multilingual_stoplist', '12.');
  ctx_ddl.add_stopword('multilingual_stoplist', '13.');
  ctx_ddl.add_stopword('multilingual_stoplist', '14.');
  ctx_ddl.add_stopword('multilingual_stoplist', '15.');
  ctx_ddl.add_stopword('multilingual_stoplist', '16.');
  ctx_ddl.add_stopword('multilingual_stoplist', '17.');
  ctx_ddl.add_stopword('multilingual_stoplist', '18.');
  ctx_ddl.add_stopword('multilingual_stoplist', '19.');
  ctx_ddl.add_stopword('multilingual_stoplist', '20.');

-- アプリ固有の停止後（LLMの応答）
  ctx_ddl.add_stopword('multilingual_stoplist', '注目すべきポイント');
  ctx_ddl.add_stopword('multilingual_stoplist', '全体的な印象や特徴');
  ctx_ddl.add_stopword('multilingual_stoplist', '画像に何が写っているか');
  ctx_ddl.add_stopword('multilingual_stoplist', '画像に描かれているテキスト');
  ctx_ddl.add_stopword('multilingual_stoplist', '画像に描かれているもののカテゴリと固有の名称');
  ctx_ddl.add_stopword('multilingual_stoplist', '画像に描かれている URL、IDなどの情報');
  ctx_ddl.add_stopword('multilingual_stoplist', 'この画像にはテキストは一切含まれていません。');

  -- 英語の停止語
  ctx_ddl.add_stopword('multilingual_stoplist', 'the');
  ctx_ddl.add_stopword('multilingual_stoplist', 'and');
  ctx_ddl.add_stopword('multilingual_stoplist', 'this');
  ctx_ddl.add_stopword('multilingual_stoplist', 'that');
  ctx_ddl.add_stopword('multilingual_stoplist', 'is');
  ctx_ddl.add_stopword('multilingual_stoplist', 'are');
  ctx_ddl.add_stopword('multilingual_stoplist', 'was');
  ctx_ddl.add_stopword('multilingual_stoplist', 'were');
  ctx_ddl.add_stopword('multilingual_stoplist', 'has');
  ctx_ddl.add_stopword('multilingual_stoplist', 'have');
END;
/


-- idx_image_caption インデックスの停止語リスト変更
ALTER INDEX idx_image_caption REBUILD
PARAMETERS ('REPLACE STOPLIST multilingual_stoplist');