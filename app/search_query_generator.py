import re
import spacy
import ginza

class SearchQueryGenerator:
    def __init__(self):
        # GiNZAのモデルをロード
        self.nlp = spacy.load("ja_ginza")
        
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
        
        # URLパターンの正規表現
        self.url_pattern = re.compile(r'https?://[^\s]+')

    def generate(self, query):
        """全文検索用のクエリーを生成する関数"""
        keywords = []
        compound_parts = []
        
        # URLを抽出
        urls = self.url_pattern.findall(query)
        if urls:
            keywords.extend(urls)
            # URL部分をクエリから削除
            query = self.url_pattern.sub('', query)
        
        # 残りのクエリに対して形態素解析を実行
        doc = None
        if query.strip():
            try:
                doc = self.nlp(query)
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
            keywords.append(query)
        
        if doc is not None:
            current_compound = []
            for i, token in enumerate(doc):
                # 数字、アルファベット、記号が連続する場合の処理
                if token.pos_ in ["NUM", "NOUN", "PROPN", "PUNCT", "X", "SYM"] and (
                    token.text.isdigit() or 
                    re.match(r'^[a-zA-Z]+$', token.text) or 
                    token.text in "!@#$%^&*()_+-=[]{}|;:,.<>?/~`"
                ):
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
                                # 漢数字とアラビア数字の形式をOR条件で結合
                                keywords.append(f"({doc[i-1].text}{token.text} OR {num_text}{token.text})")
                            continue
                        if token.text not in self.formal_nouns and token.dep_ != "fixed":
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
                        if token.text in self.color_adj_to_noun:
                            keywords.append(self.color_adj_to_noun[token.text])
                        else:
                            keywords.append(token.lemma_)
                    elif token.pos_ == "VERB" and token.dep_ not in ["aux", "fixed"]:
                        # 「する」「した」を除外
                        if token.lemma_ not in ["する", "した"]:
                            keywords.append(token.text)
            
            # 最後に残っている連続部分を処理
            if current_compound:
                compound_word = "".join(current_compound)
                keywords.append(compound_word)
        
        # print("抽出されたキーワード:", keywords)
        
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
                    # 論理グループの括弧を含むキーワードかどうかを判定
                    if keyword.startswith('(') and keyword.endswith(')') and ' OR ' in keyword:
                        # 論理グループの括弧を含む場合は、そのまま追加
                        escaped_keywords.append(keyword)
                    else:
                        # それ以外の場合は、すべての特殊文字をエスケープ
                        escaped = keyword.replace('{', '\\{').replace('}', '\\}').replace('[', '\\[').replace(']', '\\]').replace('(', '\\(').replace(')', '\\)').replace('|', '\\|').replace('*', '\\*').replace('?', '\\?').replace('+', '\\+').replace('-', '\\-').replace(':', '\\:').replace('/', '\\/').replace('.', '\\.').replace('_', '\\_')
                        escaped_keywords.append(escaped)
            
            # キーワードをANDで連結（空のキーワードを除外）
            search_query = " AND ".join(escaped_keywords) if escaped_keywords else query
        
        # print("生成された検索クエリー:", search_query)
        return search_query 