# OCI Gemini 2.5 対応 TODO

- [x] `model_settings.json` に OCI Gemini 2.5 Pro / Flash / Flash-Lite を追加する。
- [x] `app/nlp_service.py` に `oci.gemini.chat` のキャプション・回答生成分岐を追加する。
- [x] Gemini 向けの OCI `GenericChatRequest` パラメータを Llama/xAI 用設定から分ける。
- [x] Gemini モデルが VLM 一覧・OCI プロバイダーフィルタ・`MLLM_MODEL_ID` 解決で扱えることをテストする。
- [x] OCI Gemini 分岐が `ChatDetails` と `GenericChatRequest` を構築することを mock でテストする。
- [x] 関連テストを `uv run` で実行して確認する。
- [x] OCI Gemini 2.5 Pro の `topK` 400 エラーを避けるため、Gemini 分岐では `top_k` を明示設定しない。
