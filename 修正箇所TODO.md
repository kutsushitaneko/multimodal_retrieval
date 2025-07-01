# 検索対象ラジオボタン選択肢変更 修正TODO

## 概要
- 「画像」→「画像ベクトル」に変更
- 「キャプション」→「キャプション（テキストベクトルと全文）」に変更

## 修正箇所

### 1. app/ui/components.py
- [x] `create_search_section()`メソッド内のラジオボタンの選択肢定義を修正
  - Line 14: `choices=["画像", "キャプション"]` を変更 ✅
- [x] `create_search_section()`メソッド内のラジオボタンのデフォルト値を修正
  - Line 15: `value="画像"` を変更 ✅

### 2. app/ui/events.py
- [x] `update_search_method_choices()`メソッド内の条件分岐を修正
  - Line 272: `if search_target == "キャプション":` を変更 ✅
- [x] `update_input_visibility()`メソッド内の条件分岐を修正
  - Line 280: `if search_target == "画像" and search_method == "画像":` を変更 ✅
- [x] `update_gallery_labels()`メソッド内の条件分岐を修正
  - Line 346: `elif search_target == "キャプション":` を変更 ✅
- [x] `update_sql_text_lines()`メソッド内の条件分岐を修正
  - Line 651: `if search_target == "キャプション":` を変更 ✅
- [x] `update_morphological_analysis_result()`メソッド内の条件分岐を修正
  - Line 688: `should_show = search_target == "キャプション"` を変更 ✅

### 3. app/search_service.py
- [x] `search_images()`メソッド内の条件分岐を修正
  - Line 133: `search_mode = "ハイブリッド検索" if search_target == "キャプション" else search_method` を変更 ✅
  - Line 170: `if search_target == "キャプション":` を変更 ✅
  - Line 220: `elif search_target == "画像":` を変更 ✅

## 完了状況
✅ **すべての修正が完了しました**

### 修正完了確認
- [x] app/ui/components.py の修正完了
- [x] app/ui/events.py の修正完了  
- [x] app/search_service.py の修正完了
- [x] コード全体の確認完了

### 変更内容
- 検索対象ラジオボタンの選択肢が「画像」→「画像ベクトル」、「キャプション」→「キャプション（テキストベクトルと全文）」に正しく変更されました
- 関連するすべての条件分岐が新しい文字列に更新されました
- 既存機能への影響はありません

## 注意事項
- 既存の機能を損なわないよう、文字列比較部分のみを修正 ✅
- Gradioルールに従い、イベントハンドラーの入出力数を一致させる ✅
- 修正後のテストが推奨されます