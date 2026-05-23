# Anthropic/OpenAI モデル更新 TODO

- [x] Anthropic 直接 API の旧モデルを現行 Vision 対応モデルへ置換する。
  - 追加/維持: `claude-opus-4-7`, `claude-opus-4-6`, `claude-opus-4-5-20251101`, `claude-sonnet-4-6`, `claude-sonnet-4-5-20250929`, `claude-haiku-4-5-20251001`
  - 削除: `claude-opus-4-0`, `claude-sonnet-4-0`, `claude-3-7-sonnet-latest`, `claude-3-5-sonnet-latest`, `claude-3-5-haiku-latest`, `claude-3-opus-latest`
- [x] OpenAI の旧モデルを現行 Vision 対応モデルへ置換する。
  - 追加/維持: `gpt-5.5`, `gpt-5.5-pro`, `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.4-nano`
  - 削除: `gpt-5-nano`, `gpt-5-mini`, `gpt-5`, `o4-mini-2025-04-16`, `o3-pro-2025-06-10`, `o3-2025-04-16`, `o3-mini-2025-01-31`, `gpt-4.1-2025-04-14`, `gpt-4.1-mini-2025-04-14`, `gpt-4.1-nano`, `gpt-4o`, `gpt-4o-mini`, `chatgpt-4o-latest`
- [x] `model_settings.json` の表示名を `{model_id}(Provider)` に統一し、OpenAI の `OpnAI` typo を解消する。
- [x] Anthropic/OpenAI の API ルーティングが既存の `anthropic.message` / `openai.reasoning` で処理できることを確認する。
- [x] Anthropic/OpenAI の provider フィルタ、Vision 対応、最大トークン設定をテストで確認する。
- [x] `uv run pytest` で関連テストを実行する。
- [x] 変更後の todo と差分を再確認する。
