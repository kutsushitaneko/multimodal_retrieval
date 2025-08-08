# model_settings.json の temperature を初期値として反映する計画

## 目的
- model_settings.json に記載された各モデルの `temperature`（または `default_temperature`）を、UI スライダーおよび `VLMService.current_vlm_settings` の初期値として反映し、一貫して推論パラメータに伝搬される状態にする。
- 後方互換性: 設定が無い場合や不正値の場合は `0.0` を採用。

## 影響範囲（編集予定ファイル）
- `app/vlm_service.py`
- `app/ui/components.py`
- `app/ui/events.py`
- （確認のみ）`app/vlm_service_factory.py`

## 仕様詳細
### 設定値の読込
- `VLMService` に `get_model_default_temperature(model_display_name: str) -> float` を追加。
  - 優先順: `default_temperature` → `temperature` → フォールバック `0.0`。
  - 返値は `float`。値域は `[0.0, 1.0]` にクランプ。
  - 数値以外/範囲外は警告ログを出し `0.0` にフォールバック。

### 現在設定の初期化
- `VLMService.get_current_vlm_settings()` 内で初回モデル決定時に
  - `current_vlm_settings["temperature"] = get_model_default_temperature(first_model)` を実施。

### UI 初期値（スライダー）
- 検索タブ: `UIComponents.create_search_vlm_settings()` の `search_vlm_temperature` の `value` を、初期選択モデルの `get_model_default_temperature(...)` に置換。
- アップロード/編集タブ: `UIComponents.create_upload_edit_section()` の `vlm_temperature` も同様に置換。

### モデル/プロバイダ変更時のスライダー同期
- `VLMService.service_provider_changed(...)` の戻り値に Temperature スライダー更新（`gr.Slider(value=...)`）を追加。
  - 現在: `(model_dropdown, max_tokens_slider, oci_region_dropdown)`
  - 変更後（例）: `(model_dropdown, temperature_slider, max_tokens_slider, oci_region_dropdown)`
- `VLMService.model_changed(...)` の戻り値にも同様に Temperature スライダー更新を追加。
  - 現在: `(max_tokens_slider, oci_region_dropdown)`
  - 変更後（例）: `(temperature_slider, max_tokens_slider, oci_region_dropdown)`
- `UIEvents.register_vlm_settings_events(...)`
  - リスナーの `outputs` に `vlm_temperature` を追加し、戻り値の個数と順序を一致させる。
- `UIEvents.register_search_vlm_settings_events(...)`
  - 同上（検索タブ専用設定）。

### current_vlm_settings の一貫性
- モデル/プロバイダ変更で UI スライダー値を更新した直後、`VLMService.update_current_vlm_settings(temperature=initial_temp)` を呼び、内部状態も同じ初期温度で更新。
  - 実装はイベント側で `then(...)` チェーン、またはハンドラー内での直接反映のどちらかで実施。

### バリデーション/後方互換
- `model_settings.json` に `temperature` が存在しない、数値でない、範囲外の場合は `0.0` を使用。
- 既存の `max_tokens` / `OCI リージョン` の挙動は変更しない（非退行）。

### デバッグ出力
- 既存のデバッグログに、初期温度および変更後の温度を追記。

## 変更のチェックリスト
- [ ] `VLMService.get_model_default_temperature()` を追加
- [ ] `VLMService.get_current_vlm_settings()` で初期温度を設定
- [ ] `VLMService.service_provider_changed()` の戻り値に温度スライダーを追加
- [ ] `VLMService.model_changed()` の戻り値に温度スライダーを追加
- [ ] `UIEvents.register_vlm_settings_events()` の `outputs` に温度スライダーを追加（入出力数整合）
- [ ] `UIEvents.register_search_vlm_settings_events()` の `outputs` に温度スライダーを追加（入出力数整合）
- [ ] `UIComponents.create_search_vlm_settings()` の温度スライダー初期値を設定ファイルから反映
- [ ] `UIComponents.create_upload_edit_section()` の温度スライダー初期値を設定ファイルから反映
- [ ] 範囲外値のクランプ・欠損時のフォールバック確認
- [ ] 既存動作（max_tokens/OCIリージョン）の非退行確認

## テスト観点（手動）
- 設定ファイルに `temperature` を設定したモデルを初期選択とした際、UI スライダー初期値が反映される。
- プロバイダ/モデル切替時にスライダー値が設定ファイルの値へ自動更新される。
- 切替直後の推論で `current_vlm_settings["temperature"]` が UI と一致している。
- `temperature` 未設定・不正値で `0.0` が適用される。


