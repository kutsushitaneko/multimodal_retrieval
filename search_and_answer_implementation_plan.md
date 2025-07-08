# 検索と回答生成の連続実行機能 実装計画

## 概要
検索ボタンとクリアボタンの間に「検索と回答生成」ボタンを配置し、このボタンが押されたときに検索処理と回答生成処理を連続して実行する機能を実装する。

## 要件
1. **UIレイアウト変更**
   - 検索ボタンとクリアボタンの間に「検索と回答生成」ボタンを配置する
   - ボタンのスタイリングは既存のボタンと一貫性を保つ

2. **機能要件**
   - 「検索と回答生成」ボタンが押されると以下の処理を連続実行：
     1. 検索ボタンが押されたときの処理（検索実行）
     2. 回答生成ボタンが押されたときの処理（回答生成）
   - 既存の検索ボタンと回答生成ボタンの動作は変更しない

3. **エラーハンドリング**
   - 検索処理が失敗した場合は回答生成処理を実行せず、エラーメッセージを表示
   - 回答生成処理が失敗した場合は適切なエラーメッセージを表示

## 修正対象ファイルと作業項目

### 1. UIコンポーネントの修正

#### 📁 `app/ui/components.py`
- [ ] **Task 1.1**: `create_search_section()`メソッドの修正
  - 検索ボタンの行に「検索と回答生成」ボタンを追加
  - ボタンの配置順序：検索ボタン → 検索と回答生成ボタン → クリアボタン → 全件表示ボタン
  - 戻り値に`search_and_answer_button`を追加

### 2. イベントハンドリングの修正

#### 📁 `app/ui/events.py`
- [ ] **Task 2.1**: `UIEvents`クラスに新しいメソッド追加
  - `register_search_and_answer_button_events()`メソッドの実装
  - 検索処理と回答生成処理を連続実行するロジック

- [ ] **Task 2.2**: 連続実行のヘルパーメソッド追加
  - `execute_search_and_answer()`メソッドの実装
  - 検索→回答生成の連続処理ロジック
  - エラーハンドリング

### 3. メインファイルの修正

#### 📁 `multimodal_retriever.py`
- [ ] **Task 3.1**: UIコンポーネント取得の修正
  - `ui_components.create_search_section()`の戻り値受け取りを修正
  - `search_and_answer_button`変数の追加

- [ ] **Task 3.2**: イベント登録の追加
  - `ui_events.register_search_and_answer_button_events()`の呼び出し追加
  - 必要なパラメータの受け渡し

## 実装の詳細設計

### 1. UIコンポーネントの変更
```python
# app/ui/components.py の create_search_section() 内
with gr.Row():
    search_button = gr.Button("検索", variant="primary")
    search_and_answer_button = gr.Button("検索と回答生成", variant="secondary")  # 新規追加
    clear_button = gr.Button("クリア")
    show_all_button = gr.Button("全件表示")

# 戻り値に search_and_answer_button を追加
return search_target, search_method, query_input, uploaded_image, uploaded_image_column, search_button, search_and_answer_button, clear_button, show_all_button, query_examples
```

### 2. イベントハンドリングの追加
```python
# app/ui/events.py 内
def register_search_and_answer_button_events(self, search_and_answer_button, query_input, uploaded_image, search_target, search_method, top_k_slider, vector_threshold, keyword_threshold, vector_gallery, keyword_gallery, filename_text, similarity_text, caption_text, state, executed_query_text, executed_sql_text, execute_query_button, pagination_row, morphological_analysis_text, reference_image_text, answer_question_input, answer_generate_button, reference_type_radio, answer_text, answer_prompt_template_dropdown):
    """検索と回答生成ボタンのイベントを登録"""
    search_and_answer_button.click(
        fn=self.execute_search_and_answer,
        inputs=[query_input, uploaded_image, search_target, search_method, top_k_slider, vector_threshold, keyword_threshold, answer_question_input, answer_prompt_template_dropdown, reference_type_radio],
        outputs=[vector_gallery, keyword_gallery, filename_text, similarity_text, caption_text, state, executed_query_text, executed_sql_text, morphological_analysis_text, reference_image_text, answer_text, answer_generate_button, reference_type_radio]
    )

def execute_search_and_answer(self, query_input, uploaded_image, search_target, search_method, top_k_slider, vector_threshold, keyword_threshold, answer_question_input, answer_prompt_template_dropdown, reference_type_radio):
    """検索と回答生成を連続実行"""
    try:
        # 1. 検索処理の実行
        search_results = self.search_service.search_images(
            query_input, uploaded_image, search_target, search_method, 
            top_k_slider, vector_threshold, keyword_threshold
        )
        
        # 2. 検索結果の検証
        if not search_results or len(search_results) < 9:  # 必要な戻り値の数をチェック
            return error_response("検索結果が取得できませんでした。")
        
        # 3. 回答生成処理の実行
        answer = self.generate_answer(
            answer_question_input, answer_prompt_template_dropdown, 
            search_results[5], reference_type_radio  # stateオブジェクト
        )
        
        # 4. 結果の統合と返却
        return combine_search_and_answer_results(search_results, answer)
        
    except Exception as e:
        return error_response(f"処理中にエラーが発生しました: {str(e)}")
```

### 3. パラメータの受け渡し
- 既存の検索ボタンと回答生成ボタンで使用されているパラメータをすべて新しいボタンでも利用
- Gradioの入出力パラメータ数が一致するよう注意深く設計

## 注意事項

### Gradioルールの遵守
1. **Event Listenersのパラメータ数確認**
   - `inputs`と`outputs`のコンポーネント数を正確に一致させる
   - 既存のイベントハンドラーを参考に、必要な全パラメータを含める

2. **コンポーネント更新の考慮**
   - 回答生成ボタンの有効/無効状態の更新
   - 参照画像テキストの更新
   - ラジオボタンの状態更新

### エラーハンドリング
1. **段階的処理**
   - 検索処理が失敗した場合は後続処理を停止
   - 各段階でのエラーメッセージを適切に返却

2. **ユーザビリティ**
   - 処理中の状態表示（進行状況の表示）
   - エラー時の分かりやすいメッセージ

## 作業進捗チェックリスト

### Phase 1: UIコンポーネント修正
- [x] `app/ui/components.py`の`create_search_section()`を修正
- [x] 新しいボタンの追加と戻り値の更新
- [x] ボタンのスタイリング確認

### Phase 2: イベントハンドリング実装
- [x] `app/ui/events.py`に`register_search_and_answer_button_events()`メソッド追加
- [x] `execute_search_and_answer()`メソッドの実装
- [x] エラーハンドリングの実装

### Phase 3: メインファイル統合
- [x] `multimodal_retriever.py`でのコンポーネント取得修正
- [x] イベント登録の追加
- [x] パラメータ受け渡しの確認

### Phase 4: テストと検証
- [x] 検索と回答生成の連続実行テスト
- [x] エラーケースのテスト
- [x] UI表示の確認
- [x] 既存機能への影響確認

### Phase 5: 最終確認
- [x] コード全体の再確認
- [x] 修正漏れや誤って変更された部分の確認
- [x] 動作確認完了

## 実装時の考慮事項

1. **パフォーマンス**
   - 連続実行による処理時間の増加
   - ユーザーへの進行状況フィードバック

2. **状態管理**
   - 検索結果のstate管理
   - 回答生成時の画像選択状態

3. **UI/UX**
   - ボタンの配置とデザイン一貫性
   - 処理中の視覚的フィードバック

4. **既存機能への影響**
   - 既存の検索ボタンと回答生成ボタンの動作維持
   - 他のUIコンポーネントとの相互作用 