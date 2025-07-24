# 「参照する情報の種類」ラジオボタン常時有効化修正計画

## 概要
現在「参照する情報の種類」ラジオボタン（reference_type_radio）は特定の条件下でdisable（interactive=False）になっています。
この動作を変更し、常にenable（interactive=True）状態にするための修正を行います。

## 修正対象ファイル
- `app/ui/events.py`

## 修正箇所一覧

### 1. ギャラリー選択時のエラーハンドリング（handle_vector_selection関数）

#### 1-1. state_dataがNoneの場合の処理（行509）
- **場所**: `app/ui/events.py` 509行目
- **現在のコード**: `interactive=False`
- **修正内容**: `interactive=True`に変更
- **状況**: ベクトル検索ギャラリー選択時、state_dataがNoneの場合

#### 1-2. インデックスが無効な場合の処理（行540）
- **場所**: `app/ui/events.py` 540行目
- **現在のコード**: `interactive=False`
- **修正内容**: `interactive=True`に変更
- **状況**: ベクトル検索ギャラリー選択時、インデックスが範囲外の場合

#### 1-3. 正常な画像選択時の処理（行622-631）
- **場所**: `app/ui/events.py` 622-631行目
- **現在のコード**: `interactive=should_enable`（条件に依存）
- **修正内容**: `interactive=True`に固定
- **状況**: ベクトル検索ギャラリーで正常に画像が選択された場合

### 2. キーワード検索ギャラリー選択時のエラーハンドリング（handle_keyword_selection関数）

#### 2-1. state_dataがNoneの場合の処理（行661）
- **場所**: `app/ui/events.py` 661行目
- **現在のコード**: `interactive=False`
- **修正内容**: `interactive=True`に変更
- **状況**: キーワード検索ギャラリー選択時、state_dataがNoneの場合

#### 2-2. 検索結果が0件の場合の処理（行696）
- **場所**: `app/ui/events.py` 696行目
- **現在のコード**: `interactive=False`
- **修正内容**: `interactive=True`に変更
- **状況**: キーワード検索ギャラリー選択時、検索結果が0件の場合

#### 2-3. インデックスが範囲外の場合の処理（行724）
- **場所**: `app/ui/events.py` 724行目
- **現在のコード**: `interactive=False`
- **修正内容**: `interactive=True`に変更
- **状況**: キーワード検索ギャラリー選択時、インデックスが範囲外の場合

#### 2-4. 正常な画像選択時の処理（行807-816）
- **場所**: `app/ui/events.py` 807-816行目
- **現在のコード**: `interactive=should_enable`（条件に依存）
- **修正内容**: `interactive=True`に固定
- **状況**: キーワード検索ギャラリーで正常に画像が選択された場合

#### 2-5. エラー発生時の処理（行839）
- **場所**: `app/ui/events.py` 839行目
- **現在のコード**: `interactive=False`
- **修正内容**: `interactive=True`に変更
- **状況**: キーワード検索ギャラリー選択時にエラーが発生した場合

### 3. 検索結果クリア処理（clear_results関数）

#### 3-1. 結果クリア時の処理（行1101）
- **場所**: `app/ui/events.py` 1101行目
- **現在のコード**: `interactive=False`
- **修正内容**: `interactive=True`に変更
- **状況**: 検索結果をクリアする際

### 4. 検索完了後の処理（execute_search_and_answer関数）

#### 4-1. 検索と回答生成連続実行時の処理（行1549-1558）
- **場所**: `app/ui/events.py` 1549-1558行目
- **現在のコード**: `interactive=should_enable`（条件に依存）
- **修正内容**: `interactive=True`に固定
- **状況**: 検索と回答生成を連続実行した際

### 5. 参照画像更新処理（update_reference_image_and_enable_answer_generation関数）

#### 5-1. 検索完了後の参照画像設定時の処理（行3119-3128）
- **場所**: `app/ui/events.py` 3119-3128行目
- **現在のコード**: `interactive=should_enable`（条件に依存）
- **修正内容**: `interactive=True`に固定
- **状況**: 検索完了後に先頭画像の情報を参照画像にセットし、ラジオボタンを更新する際

## 修正作業の手順

1. **事前準備**
   - [ ] `app/ui/events.py`のバックアップを作成
   - [ ] 修正対象箇所をマークダウンで確認

2. **修正実行**
   - [ ] 1-1: handle_vector_selection関数のstate_dataがNoneの場合の処理を修正
   - [ ] 1-2: handle_vector_selection関数のインデックス無効の場合の処理を修正
   - [ ] 1-3: handle_vector_selection関数の正常選択時の処理を修正
   - [ ] 2-1: handle_keyword_selection関数のstate_dataがNoneの場合の処理を修正
   - [ ] 2-2: handle_keyword_selection関数の検索結果0件の場合の処理を修正
   - [ ] 2-3: handle_keyword_selection関数のインデックス範囲外の場合の処理を修正
   - [ ] 2-4: handle_keyword_selection関数の正常選択時の処理を修正
   - [ ] 2-5: handle_keyword_selection関数のエラー発生時の処理を修正
   - [ ] 3-1: clear_results関数の結果クリア時の処理を修正
   - [ ] 4-1: execute_search_and_answer関数の検索完了時の処理を修正
   - [ ] 5-1: update_reference_image_and_enable_answer_generation関数の処理を修正

3. **動作確認**
   - [ ] アプリケーションが正常に起動することを確認
   - [ ] 各種エラー状況でもラジオボタンが有効であることを確認
   - [ ] 正常な検索・選択動作でもラジオボタンが有効であることを確認

## 注意事項

- `check_answer_generation_conditions`関数は回答生成ボタンの有効/無効を判定する関数で、ラジオボタンには直接影響しないため修正対象外
- 各箇所で`should_enable`変数を使用してinteractiveを動的に設定している箇所を、すべて`True`に固定する
- エラー処理やクリア処理でも一貫してラジオボタンを有効状態に保つ

## 完了確認

すべてのチェックボックスにチェックが入ったら修正完了とする。 