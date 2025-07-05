import re
import spacy
import ginza
from .global_nlp_service import get_global_nlp_service

class SearchQueryGenerator:
    def __init__(self):
        # グローバルNLPServiceを使用してspaCyモデルを取得（シングルトンパターン）
        self.nlp_service = get_global_nlp_service()
        
        # 色の形容詞変換マップ
        self.color_adj_to_noun = {
            '赤い': '赤',
            '青い': '青',
            '白い': '白',
            '黒い': '黒',
            '黄色い': '黄色',
            '緑の': '緑',
            '紫の': '紫',
            'ピンクの': 'ピンク',
            'オレンジの': 'オレンジ',
            '茶色い': '茶色',
            '灰色の': '灰色',
            'グレーの': 'グレー',
            '金の': '金',
            '銀の': '銀'
        }
        
        # 漢数字変換マップ
        self.kanji_to_number = {
            '零': '0', '〇': '0',
            '一': '1', '壱': '1',
            '二': '2', '弐': '2',
            '三': '3', '参': '3',
            '四': '4', '肆': '4',
            '五': '5', '伍': '5',
            '六': '6', '陸': '6',
            '七': '7', '漆': '7',
            '八': '8', '捌': '8',
            '九': '9', '玖': '9',
            '十': '10', '拾': '10',
            '百': '100',
            '千': '1000',
            '万': '10000',
            '億': '100000000',
            '兆': '1000000000000'
        }
        
        # 助数詞のリスト
        self.counters = {
            '人', '匹', '頭', '羽', '冊', '枚', '台', '個', '本', '杯', '階', '歳',
            '時', '分', '秒', '年', '月', '日', '週', '回', '度', '番', '号', '代'
        }
        
        # 除外する形式名詞リスト
        self.formal_nouns = {'もの', 'こと', 'の', 'ところ', 'とき', 'もん', 'やつ', 
                           'わけ', 'はず', 'ため', 'つもり', 'よう', 'の', 'ため', 
                           'ほど', 'まま', 'くらい', 'ぐらい', 'かぎり'}
        
        # 停止語リスト（sql/create_stoplist.sqlで定義されたものと同じ）
        self.stopwords = {
            # メタデータ
            '同義語','類義語','関連語句','テーマ','カテゴリ',
            # 助詞・助動詞
            'は', 'が', 'を', 'に', 'で', 'と', 'から', 'まで', 'です', 'ます', 'である', 'ですか', 'ですよ', 'ですね', 
            'でした', 'でしたか', 'でしたよ', 'でしたね', 'について', 'において', 'に関して', 'に対して', 'だ', 'でしょう', 'かもしれない',
            # 指示語
            'これ', 'それ', 'あれ', 'どれ', 'ここ', 'そこ', 'あそこ', 'どこ', 'こう', 'そう', 'どう', 'あの',
            # 一般的すぎる名詞
            'もの', 'こと', 'とき', '場合', '内容', '説明', '情報', '状況', '結果', '仕様', '使用', '使用例', 
            '利用', '利用例', '活用', '活用例', '例', '例文', '例示', '機能', '詳細', '課題', '課題点', '問題', 
            '問題点','解決', '解決策', '可能','可能性','不可能','困難','理由','意義','意味','効果','効果的',
            '効能','構成','構造','表現','カテゴリー','範疇','クラス','クラスタ','クラスター','分類',
            '累計','類型','形式','フォーマット','軽減','削減','低減','削除','増強','増大','改善','向上','推進',
            '加速','強化','原則','減速','簡潔','要約','概要','詳述','詳細','シンプル','ディテール','ディテイル',
            '構成','注意','考慮','メリデメ','メリット','デメリット','メリットデメリット','メリットデメリット','メリットデメリット',
            '表題','タイトル','題名','題材',
            # ドメイン固有の停止語（IT）
            'データ', 'システム', 'ソフトウェア', 'ハードウェア', 'ネットワーク', 'セキュリティ', 'データベース', 
            'アプリ', 'アプリケーション', 'ツール', 'サービス', 'ソリューション', 'パラメータ', 'パラメーター', 
            'コンピュータ', 'コンピューター', 'サーバ', 'サーバー', 'スマホ', 'スマートフォン', 'モバイル', 
            'モバイルアプリ', 'モバイルアプリケーション', 'デバイス', 'コード', 'プログラム', 'サンプル', 'アーキテクチャ',
            'アーキ','パターン','実装','プログラミング','コーディング',
            # マークダウンなど
            '###', '**', '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.',
            '11.', '12.', '13.', '14.', '15.', '16.', '17.', '18.', '19.', '20.',
            # SNS関連記号
            '@',
            # 英語の停止語
            'the', 'and', 'this', 'that', 'is', 'are', 'was', 'were', 'has', 'have'
        }
        
        # 基本的な停止語（単独で意味を持たない語）
        self.basic_stopwords = {
            'の', 'を', 'に', 'が', 'は', 'で', 'と', 'から', 'まで', 'より', 'へ',
            'この', 'その', 'あの', 'どの', 'これ', 'それ', 'あれ', 'どれ',
            'ここ', 'そこ', 'あそこ', 'どこ',
            'だ', 'である', 'です', 'ます', 'した', 'する'
        }
        
        # 複合語の停止語（名詞が含まれる場合は常に除去） - 方法論的表現など
        self.compound_stopwords = [
            '使い方', '利用方法', '活用方法', '使用方法', '操作方法', '設定方法',
            '導入方法', '実装方法', 'やり方', '仕方', '進め方', '考え方', '見方',
            '捉え方', '取り組み方', '方法', '手法', '手段', '手順','方法論','注意事項','注意点','考慮事項',
            '考慮点','について', 'に関して', 'に対して', 'において', 'に関する', 'に対する', 'における',
            'を説明', 'を紹介', 'を解説', 'を記述', 'を示す', 'を表示', 'を表現', 'を教えて',
            'してください', 'してみて', 'してみる', 'していく', 'している', 'します', 'しました', 
            '教えて', '説明して', '紹介して', '解説して', '記述して', '示して', '表示して', '表現して',
            # 詳細度に関する修飾語
            '具体的', '具体的な', '具体的に', '詳しく', '詳細に', 'より詳しく', 'もっと詳しく'
        ]
        
        # 文脈依存で除外する語（複合語の一部では保持、単独では除外）
        self.context_dependent_excludes = {'方式', '工程', '流れ', '過程', '段階', '方','際','点'}
        
        # URLパターンの正規表現（英数字、記号のみを含むように修正）
        self.url_pattern = re.compile(r'https?://[a-zA-Z0-9\-._~:/?#\[\]@!$&\'()*+,;=]+')
        
        # 論文IDパターン（日本語文字と隣接しないことを保証）
        self.paper_id_pattern = re.compile(r'(?<!\d)\d{4}\.\d{4,5}(?!\d)')
        
        # バージョン番号パターン
        self.version_pattern = re.compile(r'v?\d+\.\d+(?:\.\d+)*')
        
        # ファイル名パターン（拡張子付きファイル名）
        self.filename_pattern = re.compile(r'\b[a-zA-Z0-9_-]+\.[a-zA-Z0-9]+\b')
        
        # 英数記号の連続パターン（一般的な識別子など）
        self.alphanumeric_pattern = re.compile(r'\b[A-Za-z0-9_-]+\b')
        
        # メールアドレスパターン
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')

    def generate(self, query):
        """全文検索用のクエリーを生成する関数"""
        if not query.strip():
            return ""
        
        # URLを最初に処理（特殊パターンよりも優先）
        original_query = query
        
        # 簡易形態素解析で名詞が含まれているかチェック
        nlp = self.nlp_service.get_nlp()
        doc = nlp(query)
        has_nouns = any(token.pos_ in ['NOUN', 'PROPN'] for token in doc)
        
        # 複合語の停止語を先に除去（形態素解析前に処理）
        # 名詞が含まれる場合は複合語停止語を除去、名詞のみの場合はクエリが空にならないよう注意
        removed_compounds = []
        for compound_stop in self.compound_stopwords:
            # 単純な文字列置換（日本語では単語境界が複雑なため）
            if compound_stop in query:
                removed_compounds.append(compound_stop)
                query = query.replace(compound_stop, '')
        
        # 余分な空白を削除
        query = ' '.join(query.split())
        
        # クエリが空になった場合は空文字列を返す（SQL発行を防ぐ）
        if not query.strip():
            return ""
        
        keywords = []
        compound_parts = []
        
        # URLを抽出
        urls = self.url_pattern.findall(query)
        if urls:
            # すべてのURLを中カッコ完全一致検索で処理
            for url in urls:
                # 中カッコで囲んだ完全一致検索（特殊文字を含むURLも対応）
                # URL内の中カッコをエスケープしてから全体を中カッコで囲む
                escaped_content = url.replace('{', '\\{').replace('}', '\\}')
                escaped_url = f'{{{escaped_content}}}'
                keywords.append(escaped_url)
            
            # URL部分をクエリから削除
            query = self.url_pattern.sub('', query)
        
        # 論文IDパターン（arXiv ID: YYYY.NNNNN 形式）を先に処理
        arxiv_pattern = r'(?<!\d)\d{4}\.\d{4,5}(?!\d)'
        arxiv_matches = list(re.finditer(arxiv_pattern, query))
        for match in reversed(arxiv_matches):
            arxiv_id = match.group()
            # 論文IDも中カッコ完全一致検索で統一
            # 論文ID内の中カッコをエスケープしてから全体を中カッコで囲む
            escaped_content = arxiv_id.replace('{', '\\{').replace('}', '\\}')
            keywords.append(f'{{{escaped_content}}}')
            
            # クエリから削除
            query = query[:match.start()] + query[match.end():]

        # その他の特殊文字を含む識別子を検出して処理
        special_patterns = [
            # バージョン番号（v1.2.3形式）- 中カッコ完全一致検索（より具体的なパターンを先に処理）
            (r'\bv[0-9]+\.[0-9]+(?:\.[0-9]+)*\b', lambda m: f'{{{m.group().replace("{", "\\{").replace("}", "\\}")}}}'),
            
            # メールアドレスパターン - 中カッコ完全一致検索
            (r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b', lambda m: f'{{{m.group().replace("{", "\\{").replace("}", "\\}")}}}'),
            
            # アンダースコア区切りの識別子（論文ID以外）- 中カッコ完全一致検索
            (r'\b[a-zA-Z][a-zA-Z0-9]*_[a-zA-Z0-9_]+\b', lambda m: f'{{{m.group().replace("{", "\\{").replace("}", "\\}")}}}'),
            
            # ファイル名パターン（拡張子付き）- 中カッコ完全一致検索
            (r'\b[a-zA-Z][a-zA-Z0-9_-]*\.[a-zA-Z0-9]{1,4}\b', lambda m: f'{{{m.group().replace("{", "\\{").replace("}", "\\}")}}}'),
            
            # パス形式（スラッシュ区切り）- 中カッコ完全一致検索
            (r'\b[a-zA-Z0-9_-]+/[a-zA-Z0-9/_-]+\b', lambda m: f'{{{m.group().replace("{", "\\{").replace("}", "\\}")}}}'),
            
            # API関数形式（function()）- 中カッコ完全一致検索
            (r'\b[a-zA-Z][a-zA-Z0-9]*\(\)\b', lambda m: f'{{{m.group().replace("{", "\\{").replace("}", "\\}")}}}'),
            
            # キー=値形式 - 中カッコ完全一致検索
            (r'\b[a-zA-Z][a-zA-Z0-9]*=[a-zA-Z0-9]+\b', lambda m: f'{{{m.group().replace("{", "\\{").replace("}", "\\}")}}}'),
            
            # ポート番号付きホスト - 中カッコ完全一致検索
            (r'\b[a-zA-Z0-9.-]+:\d+\b', lambda m: f'{{{m.group().replace("{", "\\{").replace("}", "\\}")}}}'),
        ]
        
        # 特殊パターンを検出してキーワードに追加し、クエリから削除
        for pattern, processor in special_patterns:
            matches = list(re.finditer(pattern, query))
            for match in reversed(matches):  # 後ろから処理して位置がずれないように
                original = match.group()
                processed = processor(match)
                if processed and processed != original:
                    keywords.append(processed)
                    # マッチした部分を完全に削除（空白で置換しない）
                    query = query[:match.start()] + query[match.end():]
        
        # 残りのクエリに対して形態素解析を実行
        doc = None
        if query.strip():
            try:
                nlp = self.nlp_service.get_nlp()
                doc = nlp(query)
            except Exception as e:
                print(f"形態素解析エラー: {e}")
                doc = None
        
        # デバッグ情報を出力
        # print("形態素解析結果:")
        if doc is not None:
            for token in doc:
                # print(f"テキスト: {token.text}, 品詞: {token.pos_}, 語幹: {token.lemma_}, 依存関係: {token.dep_}")
                pass
        else:
            # print("形態素解析が失敗しました。元のクエリを使用します。")
            if query.strip():  # 空文字列でない場合のみ追加
                keywords.append(query)
        
        if doc is not None:
            current_compound = []
            for i, token in enumerate(doc):
                # 数字、アルファベット、記号が連続する場合の処理
                # ただし、数詞の場合で次のトークンが助数詞の場合は除外（後で助数詞と組み合わせて処理するため）
                if (token.pos_ in ["NUM", "NOUN", "PROPN", "PUNCT", "X", "SYM"] and (
                    token.text.isdigit() or 
                    re.match(r'^[a-zA-Z]+$', token.text) or 
                    (token.text in "!#$%^&*()_+-=[]{}|;:,.<>?/~`")  # @記号を除外
                ) and not (
                    token.pos_ == "NUM" and 
                    i + 1 < len(doc) and 
                    doc[i+1].text in self.counters
                )):
                    # 空白を保持して追加（空白で区切られた英数記号対応）
                    if current_compound and not current_compound[-1].endswith(' '):
                        current_compound.append(' ')
                    current_compound.append(token.text)
                else:
                    # この直前までの英数記号連続が途絶えた場合はここまでの英数記号列を1つの単語にまとめてキーワードへ追加
                    if current_compound:
                        compound_word = "".join(current_compound)
                        keywords.append(compound_word)
                        current_compound = []
                    
                    # 通常の処理（英数記号の連続ではない）
                    if token.pos_ in ["NOUN", "PROPN", "X", "SYM"]:
                        # 助数詞は名詞として扱わない
                        if token.text in self.counters:
                            # 前のトークンが漢数字の場合は、組み合わせて処理
                            if i > 0 and doc[i-1].pos_ == "NUM":
                                num_text = doc[i-1].text
                                for kanji, num in self.kanji_to_number.items():
                                    num_text = num_text.replace(kanji, num)
                                
                                # 重複を避けるため、元の文字列と変換後が同じ場合は単一条件にする
                                if doc[i-1].text == num_text:
                                    keywords.append(f"{doc[i-1].text}{token.text}")
                                else:
                                    # 漢数字とアラビア数字の形式をOR条件で結合
                                    keywords.append(f"({doc[i-1].text}{token.text} OR {num_text}{token.text})")
                            continue
                        
                        # 文脈に依存する停止語の処理
                        if token.text in self.context_dependent_excludes:
                            # 複合語の一部かチェック（前後のトークンとの依存関係を確認）
                            is_part_of_compound = False
                            
                            # 前のトークンとの関係をチェック
                            if i > 0:
                                prev_token = doc[i-1]
                                if (prev_token.pos_ in ["NOUN", "PROPN"] and 
                                    token.dep_ in ["compound", "nmod"] and 
                                    prev_token.text not in self.stopwords):
                                    is_part_of_compound = True
                            
                            # 後のトークンとの関係をチェック
                            if i + 1 < len(doc):
                                next_token = doc[i+1]
                                if (next_token.pos_ in ["NOUN", "PROPN"] and 
                                    next_token.dep_ in ["compound", "nmod"] and 
                                    next_token.text not in self.stopwords):
                                    is_part_of_compound = True
                            
                            # 固有名詞（人名など）の場合は保持
                            if token.pos_ == "PROPN":
                                is_part_of_compound = True
                            
                            # 複合語の一部でない場合のみ除外
                            if not is_part_of_compound:
                                continue
                        
                        if token.text not in self.formal_nouns and token.dep_ != "fixed" and token.text not in self.stopwords and token.text != '@':
                            keywords.append(token.text)
                    elif token.pos_ == "NUM":
                        if token.dep_ != "fixed":
                            # 次のトークンが助数詞でない場合のみ、単独の数字として処理
                            if i + 1 >= len(doc) or doc[i+1].text not in self.counters:
                                num_text = token.text
                                for kanji, num in self.kanji_to_number.items():
                                    num_text = num_text.replace(kanji, num)
                                keywords.append(num_text)
                    elif token.pos_ == "ADJ":
                        # 色形容詞の場合のみ、名詞に変換して採用
                        if token.text in self.color_adj_to_noun:
                            keywords.append(self.color_adj_to_noun[token.text])
                        # 色形容詞以外は全て除外
                    # VERBは全て採用しない
            
            # 最後に残っている連続部分を処理
            if current_compound:
                compound_word = "".join(current_compound)
                keywords.append(compound_word)
        
        # print("抽出されたキーワード:", keywords)
        
        # 停止語の最終フィルタリング（単語単体で停止語に含まれているものを除外）
        filtered_keywords = []
        for keyword in keywords:
            # 中カッコ完全一致検索、論理グループ、英数記号の連続は除外対象外
            if (keyword.startswith('{') and keyword.endswith('}')) or \
               (keyword.startswith('(') and keyword.endswith(')') and ' OR ' in keyword):
                filtered_keywords.append(keyword)
            # @記号は検索ノイズになるため除外
            elif keyword == '@':
                continue
            elif keyword not in self.stopwords:
                filtered_keywords.append(keyword)
        
        keywords = filtered_keywords
        
        # キーワードが空の場合、元のクエリを使用
        if not keywords:
            keywords.append(query)
        

        
        if not keywords:
            # キーワードが見つからない場合は元のクエリーをそのまま使用
            search_query = query
        else:
            # 特殊文字をエスケープ
            escaped_keywords = []
            for keyword in keywords:
                if keyword.strip():  # 空のキーワードを除外
                    # 中カッコ完全一致検索のキーワードはそのまま
                    if keyword.startswith('{') and keyword.endswith('}'):
                        escaped_keywords.append(keyword)
                    # 論理グループの括弧を含むキーワードかどうかを判定
                    elif keyword.startswith('(') and keyword.endswith(')') and ' OR ' in keyword:
                        # 論理グループの括弧を含む場合は、そのまま追加
                        escaped_keywords.append(keyword)
                    # プレースホルダーが置換済みの AND 結合された文字列の場合はそのまま
                    elif ' AND ' in keyword:
                        escaped_keywords.append(keyword)
                    else:
                        # それ以外の場合は、すべての特殊文字をエスケープ
                        escaped = keyword.replace('{', '\\{').replace('}', '\\}').replace('[', '\\[').replace(']', '\\]').replace('(', '\\(').replace(')', '\\)').replace('|', '\\|').replace('*', '\\*').replace('?', '\\?').replace('+', '\\+').replace('-', '\\-').replace(':', '\\:').replace('/', '\\/').replace('.', '\\.').replace('_', '\\_')
                        escaped_keywords.append(escaped)
            
            # キーワードをANDで連結（空のキーワードを除外）
            search_query = " AND ".join(escaped_keywords) if escaped_keywords else query
        
        # print("生成された検索クエリー:", search_query)
        return search_query
    
    def get_morphological_analysis_details(self, query):
        """形態素解析の詳細結果を取得する関数（マークダウンテーブル形式）"""
        if not query.strip():
            return ""
        
        morphological_details = []
        original_query = query  # 元のクエリを保存
        keywords = []  # 特殊文字処理で追加されるキーワード
        
        # 簡易形態素解析で名詞が含まれているかチェック
        nlp = self.nlp_service.get_nlp()
        doc = nlp(query)
        has_nouns = any(token.pos_ in ['NOUN', 'PROPN'] for token in doc)
        
        # 複合語の停止語を先に除去（形態素解析前に処理）
        # 名詞が含まれる場合は複合語停止語を除去、名詞のみの場合はクエリが空にならないよう注意
        removed_compounds = []
        for compound_stop in self.compound_stopwords:
            # 単純な文字列置換（日本語では単語境界が複雑なため）
            if compound_stop in query:
                removed_compounds.append(compound_stop)
                query = query.replace(compound_stop, '')
        
        # 余分な空白を削除
        query = ' '.join(query.split())
        
        # 複合語停止語の除去結果を表示
        if removed_compounds:
            morphological_details.append("### 🚫 複合語停止語の除去")
            morphological_details.append("**処理概要:** 名詞・固有名詞が含まれるクエリで、方法論的表現を自動除去")
            morphological_details.append("**効果:** 検索範囲を拡大し、「使い方」「説明」などの記載がない文書も検索対象に")
            morphological_details.append("")
            morphological_details.append("**除去された表現:**")
            for compound in removed_compounds:
                morphological_details.append(f"- `{compound}` （検索ノイズとなるため除去）")
            morphological_details.append("")
        
        # URLを先に処理（特殊パターンよりも優先）
        # URLを抽出
        urls = self.url_pattern.findall(query)
        if urls:
            morphological_details.append("### 🔗 URL検出と処理")
            for i, url in enumerate(urls):
                morphological_details.append(f"- **URL {i+1}:** `{url}`")
                morphological_details.append(f"- **中カッコ完全一致検索:** `{{{url}}}`")
                morphological_details.append(f"- **理由:** Oracle Textの中カッコ機能で特殊文字を含むURLも完全一致検索可能")
            morphological_details.append("")  # 空行を追加
            # URL部分をクエリから削除
            query = self.url_pattern.sub('', query)
        
        # 論文IDパターン（arXiv ID: YYYY.NNNNN 形式）を先に処理
        arxiv_pattern = r'(?<!\d)\d{4}\.\d{4,5}(?!\d)'
        arxiv_matches = list(re.finditer(arxiv_pattern, query))
        if arxiv_matches:
            morphological_details.append("### 📄 論文ID検出と処理")
            for i, match in enumerate(reversed(arxiv_matches)):
                arxiv_id = match.group()
                morphological_details.append(f"- **論文ID {i+1}:** `{arxiv_id}`")
                morphological_details.append(f"- **中カッコ完全一致検索:** `{{{arxiv_id}}}`")
                morphological_details.append(f"- **理由:** Oracle Textの中カッコ機能で論文IDも他の識別子と統一して正確に検索")
                keywords.append(f'{{{arxiv_id}}}')
                # クエリから削除（後ろから処理）
                query = query[:match.start()] + query[match.end():]
            morphological_details.append("")

        # その他の特殊文字を含む識別子を検出して処理
        special_patterns = [
            # バージョン番号（v1.2.3形式）- 中カッコ完全一致検索（より具体的なパターンを先に処理）
            (r'\bv[0-9]+\.[0-9]+(?:\.[0-9]+)*\b', lambda m: f'{{{m.group().replace("{", "\\{").replace("}", "\\}")}}}'),
            
            # メールアドレスパターン - 中カッコ完全一致検索
            (r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b', lambda m: f'{{{m.group().replace("{", "\\{").replace("}", "\\}")}}}'),
            
            # アンダースコア区切りの識別子（論文ID以外）- 中カッコ完全一致検索
            (r'\b[a-zA-Z][a-zA-Z0-9]*_[a-zA-Z0-9_]+\b', lambda m: f'{{{m.group().replace("{", "\\{").replace("}", "\\}")}}}'),
            
            # ファイル名パターン（拡張子付き）- 中カッコ完全一致検索
            (r'\b[a-zA-Z][a-zA-Z0-9_-]*\.[a-zA-Z0-9]{1,4}\b', lambda m: f'{{{m.group().replace("{", "\\{").replace("}", "\\}")}}}'),
            
            # パス形式（スラッシュ区切り）- 中カッコ完全一致検索
            (r'\b[a-zA-Z0-9_-]+/[a-zA-Z0-9/_-]+\b', lambda m: f'{{{m.group().replace("{", "\\{").replace("}", "\\}")}}}'),
            
            # API関数形式（function()）- 中カッコ完全一致検索
            (r'\b[a-zA-Z][a-zA-Z0-9]*\(\)\b', lambda m: f'{{{m.group().replace("{", "\\{").replace("}", "\\}")}}}'),
            
            # キー=値形式 - 中カッコ完全一致検索
            (r'\b[a-zA-Z][a-zA-Z0-9]*=[a-zA-Z0-9]+\b', lambda m: f'{{{m.group().replace("{", "\\{").replace("}", "\\}")}}}'),
            
            # ポート番号付きホスト - 中カッコ完全一致検索
            (r'\b[a-zA-Z0-9.-]+:\d+\b', lambda m: f'{{{m.group().replace("{", "\\{").replace("}", "\\}")}}}'),
        ]
        
        # 特殊パターンを検出してキーワードに追加し、クエリから削除
        for pattern, processor in special_patterns:
            matches = list(re.finditer(pattern, query))
            for match in reversed(matches):
                original = match.group()
                processed = processor(match)
                if processed and processed != original:
                    keywords.append(processed)
                    morphological_details.append(f"### 🔧 特殊文字処理")
                    morphological_details.append(f"- **検出:** `{original}`")
                    morphological_details.append(f"- **変換:** `{processed}`")
                    morphological_details.append(f"- **理由:** Oracle Textの中カッコ完全一致検索で特殊文字を含む識別子も正確に検索")
                    morphological_details.append("")
                    query = query[:match.start()] + query[match.end():]
        
        # 残りのクエリに対して形態素解析を実行
        doc = None
        if query.strip():
            try:
                nlp = self.nlp_service.get_nlp()
                doc = nlp(query)
            except Exception as e:
                morphological_details.append(f"❌ **形態素解析エラー:** `{e}`")
                return "\n".join(morphological_details)
        
        if doc is not None:
            morphological_details.append("### 📝 形態素解析結果")
            morphological_details.append("")
            
            # テーブルヘッダー
            morphological_details.append("| # | 単語 | 品詞 | 原形 | 依存関係 | 処理状況 |")
            morphological_details.append("|---|------|------|------|----------|----------|")
            
            for i, token in enumerate(doc):
                # 処理状況を判定（generateメソッドの実際のロジックに合わせる）
                status = "❌ 除外"
                adopted = False
                
                # 英数記号連続の場合の判定
                is_alpha_num_symbol = (token.pos_ in ["NUM", "NOUN", "PROPN", "PUNCT", "X", "SYM"] and (
                    token.text.isdigit() or 
                    re.match(r'^[a-zA-Z]+$', token.text) or 
                    token.text in "!@#$%^&*()_+-=[]{}|;:,.<>?/~`"
                ))
                
                if is_alpha_num_symbol:
                    status = "✅ 英数記号連続として採用"
                    adopted = True
                elif token.pos_ in ["NOUN", "PROPN", "X", "SYM"]:
                    # 助数詞チェック
                    if token.text in self.counters:
                        # 前のトークンが数詞の場合の特殊処理
                        if i > 0 and doc[i-1].pos_ == "NUM":
                            status = "📊 助数詞（数詞と組み合わせ）"
                            adopted = True
                        else:
                            status = "❌ 助数詞のため除外"
                    # 文脈に依存する停止語の処理
                    elif token.text in self.context_dependent_excludes:
                        # 複合語の一部かチェック（前後のトークンとの依存関係を確認）
                        is_part_of_compound = False
                        
                        # 前のトークンとの関係をチェック
                        if i > 0:
                            prev_token = doc[i-1]
                            if (prev_token.pos_ in ["NOUN", "PROPN"] and 
                                token.dep_ in ["compound", "nmod"] and 
                                prev_token.text not in self.stopwords):
                                is_part_of_compound = True
                        
                        # 後のトークンとの関係をチェック
                        if i + 1 < len(doc):
                            next_token = doc[i+1]
                            if (next_token.pos_ in ["NOUN", "PROPN"] and 
                                next_token.dep_ in ["compound", "nmod"] and 
                                next_token.text not in self.stopwords):
                                is_part_of_compound = True
                        
                        # 固有名詞（人名など）の場合は保持
                        if token.pos_ == "PROPN":
                            is_part_of_compound = True
                        
                        # 複合語の一部でない場合のみ除外
                        if not is_part_of_compound:
                            status = "🚫 文脈依存停止語のため除外"
                        else:
                            status = "✅ 複合語の一部として採用"
                            adopted = True
                    # 停止語チェック
                    elif token.text in self.stopwords:
                        status = "⛔ 停止語のため除外"
                    # 形式名詞チェック
                    elif token.text in self.formal_nouns:
                        status = "❌ 形式名詞のため除外"
                    # fixed依存関係チェック
                    elif token.dep_ == "fixed":
                        status = "❌ fixed依存のため除外"
                    else:
                        status = "✅ 採用"
                        adopted = True
                elif token.pos_ == "NUM":
                    if token.dep_ == "fixed":
                        status = "❌ fixed依存のため除外"
                    else:
                        # 次のトークンが助数詞でない場合のみ
                        if i + 1 >= len(doc) or doc[i+1].text not in self.counters:
                            status = "✅ 採用"
                            adopted = True
                        else:
                            status = "📊 助数詞と組み合わせ"
                elif token.pos_ == "ADJ":
                    if token.text in self.color_adj_to_noun:
                        status = f"✅ 採用（色形容詞 → `{self.color_adj_to_noun[token.text]}`に変換）"
                        adopted = True
                    else:
                        status = "❌ 色形容詞以外のため除外"
                elif token.pos_ == "VERB":
                    status = "❌ 動詞のため除外"
                else:
                    status = "❌ 対象外品詞"
                
                # テーブル行を作成
                row = f"| {i+1:2d} | `{token.text}` | {token.pos_} | `{token.lemma_}` | {token.dep_} | {status} |"
                morphological_details.append(row)
            
            # 品詞の説明を追加
            morphological_details.append("")
            morphological_details.append("#### 📚 品詞の説明（一部）")
            morphological_details.append("- **ADJ**: 形容詞")
            morphological_details.append("- **ADP**: 助詞")
            morphological_details.append("- **ADV**: 副詞")
            morphological_details.append("- **ADX**: 助動詞")
            morphological_details.append("- **NOUN**: 名詞")
            morphological_details.append("- **NUM**: 数詞")
            morphological_details.append("- **PART**: 終助詞・間投助詞")
            morphological_details.append("- **PRON**: 代名詞")
            morphological_details.append("- **PROPN**: 固有名詞")
            morphological_details.append("- **PUNCT**: 句読点")
            morphological_details.append("- **SCONJ **: 句読点")
            morphological_details.append("- **SYM**: 記号") 
            morphological_details.append("- **VERB**: 動詞")
            morphological_details.append("- **X**: その他、未知語")           
            morphological_details.append("")
            morphological_details.append("#### ⛔ 停止語フィルタリング")
            morphological_details.append("- **停止語**: 一般的すぎてノイズとなる語")
            morphological_details.append("- 例） '説明', '使い方', '注意', '考慮', 'メリデメ', 'システム', 'データ' など")
        
        elif query.strip():
            morphological_details.append("### 📝 形態素解析結果")
            morphological_details.append("❌ **形態素解析が失敗しました。**")
            morphological_details.append("- URL以外の部分の処理でエラーが発生しました")
        
        # 最終的な検索クエリの表示
        if urls or doc is not None:
            morphological_details.append("")
            morphological_details.append("#### 🔍 最終検索クエリ")
            final_query = self.generate(original_query)
            morphological_details.append(f"```")
            morphological_details.append(f"{final_query}")
            morphological_details.append(f"```")
        
        return "\n".join(morphological_details) if morphological_details else ""

 