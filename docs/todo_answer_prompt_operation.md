# 回答プロンプト Material/Operation 趣旨（平易日本語）反映 TODO

## 修正箇所
- [x] `prompt/answer/デフォルト（回答生成）.txt` — 第1段落を推奨案に差し替え（「翻訳以外」例外削除）
- [x] `prompt/answer/ハルシネーション防止版回答生成.txt` — 同趣旨を当てはめ
- [x] プロンプト文言を参照するテストがあれば更新（該当なし）

## 確認
- [x] 全文の整合（`{documents}` / `{query_text}` プレースホルダ維持）
- [x] 関連テスト実行（workflow + react: 89 passed）
