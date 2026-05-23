# DB接続リファクタ チェックリスト

## 方針
- プロセス内単一プール（`pool_alias="multimodal_retriever"`）
- `ping_interval` / `ping_timeout` による死活管理
- 接続エラー時は `pool.drop(conn)` + リトライ（プール全体の close/再作成はしない）
- 自前ヘルスチェックスレッド廃止

## 実装タスク

- [x] Step1: `app/config.py` — プール集約・不要メソッド削除
- [x] Step1 テスト: `tests/test_config_db_pool.py`
- [x] Step2: `multimodal_retriever.py` — 監視スレッド削除・終了処理
- [x] Step2 テスト: `tests/test_multimodal_retriever_db.py`
- [x] Step3: `app/database_service.py` — `_execute_with_retry` 改善・operation(conn) 化
- [x] Step3 テスト: `tests/test_database_service_retry.py`
- [x] Step4: `app/ui/events.py` — 共有 DatabaseService 利用
- [x] Step4 テスト: `tests/test_events_database_service.py`
- [x] Step5: `docs/app_classes.md` 更新
- [x] 最終確認: 全 unit テスト実行

## 手動テスト（実装後）

| # | 手順 | 期待結果 |
|---|------|----------|
| 1 | `uv run python multimodal_retriever.py` で起動 | エラーなく起動し、初期表示で画像一覧が読み込まれる |
| 2 | 「検索と回答生成」でベクトル検索 | 検索結果がギャラリーに表示される |
| 3 | 全文検索（形態素解析あり） | キーワード検索結果が表示される |
| 4 | 「イメージ管理」で新規画像を登録 | 「データベースに登録しました」と表示される |
| 5 | 登録済み画像のキャプション更新 | 更新成功メッセージが表示される |
| 6 | 画像削除（確認チェック後） | 削除成功し、DB から消える |
| 7 | **DB を停止** → 検索を実行 | エラーが表示される（アプリは落ちなくてよい） |
| 8 | **DB を再起動** → 30〜60 秒待機 → 再検索 | **プロセス再起動なし**で検索が成功する |
| 9 | 再起動後に新規登録 | 登録が成功する（共有プール経由） |
| 10 | アプリ終了（Ctrl+C） | 異常終了せず終了する |
