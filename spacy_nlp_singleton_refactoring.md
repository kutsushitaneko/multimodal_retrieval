# spaCy ja_ginzaモデル グローバルシングルトン化 修正計画

## 修正の目的
- メモリ使用量の削減（複数のNLPServiceインスタンス → 単一のグローバルインスタンス）
- ja_ginzaモデルの初期化を1回のみに制限
- マルチセッション環境での安全性確保

## 修正箇所一覧

### 1. グローバルシングルトンNLPServiceの作成
- [x] `app/global_nlp_service.py` - 新規作成
  - グローバルシングルトンNLPServiceクラスの実装
  - スレッドセーフな初期化
  - モジュールレベルでのインスタンス管理

### 2. 既存のNLPServiceインスタンス作成箇所の修正

#### `app/search_query_generator.py`
- [x] Line 8: `self.nlp_service = NLPService()` → グローバルインスタンス使用に変更

#### `app/ui/events.py`
- [x] Line 30: `self.search_nlp_service = NLPService(self.search_vlm_service)` → 修正
- [x] Line 31: `self.upload_nlp_service = NLPService(self.upload_vlm_service)` → 修正
- [x] Line 2259: `nlp_service = NLPService()` → 修正
- [x] Line 2311: `nlp_service = NLPService()`→ 修正

### 3. VLMService依存関係の調整
- [x] NLPServiceとVLMServiceの分離
- [x] VLMServiceの独立した管理

### 4. インポート文の修正
- [x] 各ファイルでのインポート文をグローバルシングルトンに変更

## 実装戦略
1. グローバルシングルトンNLPServiceを作成
2. 既存のNLPServiceインスタンス作成を段階的に置換
3. VLMServiceは独立して管理
4. 修正後の動作確認

## 期待される効果
- メモリ使用量の大幅削減
- 初期化時間の短縮
- システム全体のパフォーマンス向上
- マルチセッション環境での安定性確保

## 修正完了確認

### ✅ 動作確認テスト結果
1. **グローバルNLPServiceの初期化**: 成功
2. **spaCyモデルの初期化**: 成功
3. **SearchQueryGeneratorの動作**: 成功（"赤い車" → "赤 AND 車"）

### ✅ 修正内容まとめ
1. **新規作成**: `app/global_nlp_service.py`
   - グローバルシングルトンNLPServiceクラス
   - スレッドセーフな初期化機能
   - モジュールレベルでのインスタンス管理

2. **修正完了**: `app/search_query_generator.py`
   - グローバルNLPServiceの使用に変更

3. **修正完了**: `app/ui/events.py`
   - グローバルNLPServiceの使用に変更
   - VLMServiceとの分離
   - 各タブで独立したVLMService使用

### ✅ 技術的改善点
- **メモリ効率**: 複数のNLPServiceインスタンス → 単一のグローバルインスタンス
- **初期化コスト**: ja_ginzaモデルの初期化を1回のみに制限
- **マルチセッション対応**: スレッドセーフな実装でマルチユーザー環境に対応
- **VLM設定独立**: 検索タブとアップロードタブで独立したVLM設定を維持 