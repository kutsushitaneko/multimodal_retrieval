# 検索ボタンを押したときに回答欄をクリアする修正計画

## 修正対象

### 1. `app/ui/events.py` の `clear_before_search` 関数
- [x] 引数に `answer_text` パラメータを追加
- [x] 戻り値に空文字の `answer_text` を追加

### 2. `app/ui/events.py` の `register_search_button_events` 関数  
- [x] 引数に `answer_text` パラメータを追加
- [x] `clear_before_search` の呼び出しで `answer_text` を `inputs` と `outputs` に含める（2箇所）

### 3. `multimodal_retriever.py` のイベント登録
- [x] `register_search_button_events` の呼び出しに `answer_text` 引数を追加

## 修正詳細

### 修正箇所1: `clear_before_search` 関数（1016行付近）
現在の関数シグネチャ:
```python
def clear_before_search(self, reference_image_text=None):
```

修正後:
```python
def clear_before_search(self, reference_image_text=None, answer_text=None):
```

### 修正箇所2: `register_search_button_events` 関数（182行付近）
引数に `answer_text=None` を追加し、`clear_before_search` の呼び出し2箇所を修正

### 修正箇所3: `multimodal_retriever.py` のイベント登録
`answer_text` を引数として渡すように修正

## 修正完了の確認

すべての修正項目が完了しました。

### 確認事項
1. ✅ `clear_before_search` 関数に `answer_text` パラメータが追加され、戻り値にも空文字の `answer_text` が含まれます
2. ✅ `register_search_button_events` 関数の引数に `answer_text` が追加され、`clear_before_search` の呼び出しでも適切に処理されます
3. ✅ `multimodal_retriever.py` での呼び出しに `answer_text` 引数が追加されました

これで、検索ボタンを押したときに「自然言語による回答」の「回答」欄がクリアされるようになります。 