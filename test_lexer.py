#!/usr/bin/env python3
"""
Oracle Text レクサーの動作検証スクリプト
"""

import oracledb
import os
from dotenv import load_dotenv

def test_oracle_text_lexer():
    """Oracle Textレクサーの動作を検証"""
    load_dotenv()
    
    # データベース接続
    db_connection = oracledb.connect(
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        dsn=os.getenv('DB_DSN')
    )
    
    try:
        cursor = db_connection.cursor()
        
        print("=== Oracle Text レクサー動作検証 ===\n")
        
        # テストケース定義
        test_cases = [
            # 論文ID関連
            "2401.05856",
            "arXiv:2401.05856",
            "https://arxiv.org/abs/2401.05856",
            
            # アンダースコア問題
            "search_queries_only",
            "user_name",
            "API_KEY",
            
            # URL関連
            "https://x.com/yuji_amanagawa",
            "https://qiita.com/yuji-arakawa",
            "https://qiita.com/yuji-arakawa/items/28f30a5434ba429f3f16",
            
            # ファイル名
            "config.json",
            "script.py",
            "my_file.txt",
            
            # バージョン番号
            "v1.2.3",
            "2.0.1",
            
            # 日本語との組み合わせ
            "設定ファイル config.json",
            "ユーザー名 user_name"
        ]
        
        for i, test_query in enumerate(test_cases, 1):
            print(f"--- テストケース {i}: '{test_query}' ---")
            
            # 1. LIKE検索でデータが存在するかチェック
            cursor.execute("""
                SELECT COUNT(*) as count
                FROM IMAGES 
                WHERE caption LIKE :1
            """, [f'%{test_query}%'])
            
            like_count = cursor.fetchone()[0]
            print(f"LIKE検索結果: {like_count}件")
            
            if like_count > 0:
                # 2. Oracle Text検索テスト（複数パターン）
                search_patterns = [
                    test_query,                    # そのまま
                    f'"{test_query}"',            # 引用符で囲む
                ]
                
                # アンダースコアがある場合は分割パターンも追加
                if '_' in test_query:
                    parts = test_query.split('_')
                    search_patterns.append(' AND '.join(parts))
                
                # ピリオドがある場合は分割パターンも追加
                if '.' in test_query and not test_query.startswith('http'):
                    parts = test_query.replace('.', ' ').split()
                    if len(parts) > 1:
                        search_patterns.append(' AND '.join(parts))
                
                for pattern in search_patterns:
                    try:
                        cursor.execute("""
                            SELECT COUNT(*) as count
                            FROM IMAGES 
                            WHERE CONTAINS(caption, :1, 1) > 0
                        """, [pattern])
                        
                        oracle_count = cursor.fetchone()[0]
                        status = "✅ 成功" if oracle_count > 0 else "❌ 失敗"
                        print(f"  Oracle Text '{pattern}': {oracle_count}件 {status}")
                        
                    except Exception as e:
                        print(f"  Oracle Text '{pattern}': エラー - {e}")
                
                # 3. 実際にヒットしたレコードからトークン分析（最初の1件のみ）
                if like_count > 0:
                    cursor.execute("""
                        SELECT image_id, SUBSTR(caption, 1, 100) as caption_preview
                        FROM IMAGES 
                        WHERE caption LIKE :1
                        AND ROWNUM = 1
                    """, [f'%{test_query}%'])
                    
                    result = cursor.fetchone()
                    if result:
                        image_id, caption_preview = result
                        print(f"  サンプルレコード ID:{image_id}")
                        print(f"  キャプション: {caption_preview}...")
                        
                        # トークン分析
                        try:
                            cursor.execute("""
                                SELECT TOKEN_TEXT, TOKEN_TYPE, TOKEN_FIRST, TOKEN_LAST
                                FROM CTX_DOC.TOKENS('IDX_IMAGE_CAPTION', :1, 1)
                                WHERE UPPER(TOKEN_TEXT) LIKE UPPER(:2)
                                ORDER BY TOKEN_FIRST
                            """, [image_id, f'%{test_query.replace("_", "%").replace(".", "%")}%'])
                            
                            tokens = cursor.fetchall()
                            if tokens:
                                print(f"  関連トークン:")
                                for token in tokens:
                                    print(f"    '{token[0]}' (タイプ:{token[1]}, 位置:{token[2]}-{token[3]})")
                            else:
                                print(f"  関連トークンなし（レクサーで分割されている可能性）")
                                
                        except Exception as e:
                            print(f"  トークン分析エラー: {e}")
            else:
                print("  対象データなし - スキップ")
            
            print()
        
        # レクサー設定の確認
        print("=== レクサー設定確認 ===")
        cursor.execute("""
            SELECT idx_name, idx_lexer
            FROM ctx_user_indexes 
            WHERE idx_name = 'IDX_IMAGE_CAPTION'
        """)
        
        result = cursor.fetchone()
        if result:
            print(f"インデックス名: {result[0]}")
            print(f"レクサー: {result[1]}")
        
        # レクサー詳細設定
        cursor.execute("""
            SELECT pre_name, pre_object 
            FROM ctx_user_preferences 
            WHERE pre_name LIKE '%lexerpref4japanese%'
        """)
        
        prefs = cursor.fetchall()
        if prefs:
            print("レクサー設定:")
            for pref in prefs:
                print(f"  {pref[0]}: {pref[1]}")
        
    finally:
        cursor.close()
        db_connection.close()

if __name__ == "__main__":
    test_oracle_text_lexer() 