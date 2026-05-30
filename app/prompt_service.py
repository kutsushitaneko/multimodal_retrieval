import os
import glob
from typing import List, Optional

from app.paths import PROMPT_ANSWER_DIR, PROMPT_CAPTION_DIR


class PromptService:
    """プロンプトテンプレートを管理するサービス"""

    CATEGORIES = {
        "caption": (PROMPT_CAPTION_DIR, "デフォルト"),
        "answer": (PROMPT_ANSWER_DIR, "デフォルト（回答生成）"),
    }

    def __init__(self, category: str = "caption"):
        if category not in self.CATEGORIES:
            raise ValueError(f"未知のプロンプトカテゴリ: {category}")
        self.category = category
        self.prompt_dir, self.default_template_name = self.CATEGORIES[category]
        self._ensure_prompt_directory()

    def _ensure_prompt_directory(self):
        """プロンプトディレクトリが存在することを確認"""
        if not os.path.exists(self.prompt_dir):
            os.makedirs(self.prompt_dir)

    def get_template_names(self) -> List[str]:
        """利用可能なプロンプトテンプレート名のリストを取得"""
        pattern = os.path.join(self.prompt_dir, "*.txt")
        template_files = glob.glob(pattern)
        template_names = []

        for file_path in template_files:
            filename = os.path.basename(file_path)
            template_name = os.path.splitext(filename)[0]
            template_names.append(template_name)

        if self.default_template_name in template_names:
            template_names.remove(self.default_template_name)
            template_names.insert(0, self.default_template_name)

        return template_names

    def load_template(self, template_name: str) -> Optional[str]:
        """指定されたテンプレート名のプロンプトを読み込み"""
        if not template_name:
            return None

        file_path = os.path.join(self.prompt_dir, f"{template_name}.txt")

        if not os.path.exists(file_path):
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            print(f"プロンプトテンプレートの読み込みエラー: {e}")
            return None

    def save_template(self, template_name: str, prompt_text: str) -> bool:
        """プロンプトテンプレートを保存"""
        if not template_name or not prompt_text:
            return False

        if template_name == self.default_template_name:
            print(f"エラー: '{self.default_template_name}'テンプレートは上書きできません")
            return False

        invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
        if any(char in template_name for char in invalid_chars):
            print(f"エラー: テンプレート名に使用できない文字が含まれています: {template_name}")
            return False

        file_path = os.path.join(self.prompt_dir, f"{template_name}.txt")

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(prompt_text)
            print(f"プロンプトテンプレート '{template_name}' を保存しました")
            return True
        except Exception as e:
            print(f"プロンプトテンプレートの保存エラー: {e}")
            return False

    def delete_template(self, template_name: str) -> bool:
        """プロンプトテンプレートを削除"""
        if not template_name:
            return False

        if template_name == self.default_template_name:
            print(f"エラー: '{self.default_template_name}'テンプレートは削除できません")
            return False

        file_path = os.path.join(self.prompt_dir, f"{template_name}.txt")

        if not os.path.exists(file_path):
            print(f"エラー: テンプレート '{template_name}' が見つかりません")
            return False

        try:
            os.remove(file_path)
            print(f"プロンプトテンプレート '{template_name}' を削除しました")
            return True
        except Exception as e:
            print(f"プロンプトテンプレートの削除エラー: {e}")
            return False

    def get_default_template_name(self) -> str:
        """デフォルトテンプレート名を取得"""
        return self.default_template_name

    @staticmethod
    def render_answer_prompt(template: str, query_text: str, documents: str) -> str:
        """回答生成プロンプトのプレースホルダを置換する"""
        if not template:
            return template
        normalized = template.replace("{{query_text}}", "{query_text}").replace("{{documents}}", "{documents}")
        rendered = normalized.replace("{query_text}", query_text or "").replace("{documents}", documents or "")
        if "{query_text}" not in normalized and "{documents}" not in normalized:
            rendered = f"{rendered}\n\n質問:\n{query_text}\n\n参照情報:\n{documents}"
        return rendered
