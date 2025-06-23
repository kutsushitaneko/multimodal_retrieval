#!/usr/bin/env python3
"""
修正後の検索クエリ生成器のテストスクリプト
"""

import sys
sys.path.append('.')
from app.search_query_generator import SearchQueryGenerator

def test_improved_search_generator():
    """修正後の検索クエリ生成器をテスト"""
    generator = SearchQueryGenerator()
    
    print("=== 修正後の検索クエリ生成器テスト ===\n")
    
    # テストケース定義
    test_cases = [
        # 論文ID（修正対象）
        "2401.05856",
        "arXiv:2401.05856について",
        "論文 2312.10997 を読んだ",
        
        # アンダースコア（問題の確認）
        "API_KEY",
        "search_queries_only",
        "user_name の設定",
        
        # URL（既存動作の確認）
        "https://x.com/yuji_amanagawa",
        "https://qiita.com/yuji-arakawa/items/28f30a5434ba429f3f16",
        
        # 複合パターン
        "2401.05856 と API_KEY の関係",
        "v1.2.3 のバージョン",
        "config.json ファイル",
        
        # 以前問題だったパターン
        "search_queries_only の分析結果",
    ]
    
    for i, test_query in enumerate(test_cases, 1):
        print(f"--- テストケース {i}: '{test_query}' ---")
        
        # 検索クエリ生成
        search_query = generator.generate(test_query)
        print(f"生成クエリ: {search_query}")
        
        # 詳細分析も表示
        details = generator.get_morphological_analysis_details(test_query)
        if details:
            print("詳細分析:")
            # マークダウンの見出しを簡略化して表示
            for line in details.split('\n'):
                if line.startswith('### '):
                    print(f"  {line[4:]}")
                elif line.startswith('- **'):
                    print(f"    {line}")
                elif line.strip() and not line.startswith('|') and not line.startswith('#'):
                    print(f"    {line}")
        
        print()

if __name__ == "__main__":
    test_improved_search_generator() 