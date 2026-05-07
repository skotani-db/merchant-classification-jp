import re


nDAY = r'(?:[0-3]?\d)'  # 日は1から31まで（先頭のゼロあり）

nMNTH = r'(?:11|12|10|0?[1-9])' # 月は1から12まで（先頭のゼロあり）

nYR = r'(?:(?:19|20)\d\d)'  # 年は20世紀または21世紀に限定している

nDELIM = r'(?:[\/\-\._])?' 

NUM_DATE = f"""
    (?P<num_date>
        (?:^|\D) # 新しいビット
        (?:
        # YYYY-MM-DD
        (?:{nYR}(?P<delim1>[\/\-\._]?){nMNTH}(?P=delim1){nDAY})
        |
        # YYYY-DD-MM
        (?:{nYR}(?P<delim2>[\/\-\._]?){nDAY}(?P=delim2){nMNTH})
        |
        # DD-MM-YYYY
        (?:{nDAY}(?P<delim3>[\/\-\._]?){nMNTH}(?P=delim3){nYR})
        |
        # MM-DD-YYYY
        (?:{nMNTH}(?P<delim4>[\/\-\._]?){nDAY}(?P=delim4){nYR})
        )
        (?:\D|$) # 新しいビット
    )"""

DAY = r"""
(?:
    # 1st 2nd 3rd など、または first second third を検索
    (?:[23]?1st|2{1,2}nd|\d{1,2}th|2?3rd|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth)
    |
    # または単なる数字
    (?:[0123]?\d)
)"""

MONTH = r'(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)'

YEAR = r"""(?:(?:[12]?\d|')?\d\d)"""

DELIM = r'(?:\s*(?:[\s\.\-\\/,]|(?:of))\s*)'

YEAR_4D = r"""(?:[12]\d\d\d)"""

DATE_PATTERN = f"""(?P<wordy_date>
    # 非単語文字または文字列の先頭
    (?:^|\W)
        (?:
            # 年・月・日のさまざまな組み合わせにマッチ
            (?:
                # 4桁の年
                (?:{YEAR_4D}{DELIM})?
                    (?:
                    # 日 - 月
                    (?:{DAY}{DELIM}{MONTH})
                    |
                    # 月 - 日
                    (?:{MONTH}{DELIM}{DAY})
                    )
                # 2桁または4桁の年
                (?:{DELIM}{YEAR})?
            )
            |
            # 月 - 年（2桁または3桁）
            (?:{MONTH}{DELIM}{YEAR})
            # 区切りなしの日付
            |
            (?:{DAY}{MONTH}{YEAR})
            |
            (?:{DAY}{MONTH}{YEAR_4D})
            |
            (?:xx{DELIM}xx{DELIM}{YEAR_4D})
        )
    # 非単語文字または文字列の末尾
    (?:$|\W)
)"""

TIME = r"""(?:
(?:
# 最初の数字は0から59まで（任意の先頭ゼロあり）
[012345]?\d
# 2番目の数字はコロン・ドット・hの後に同じ形式で続く
(:|\.|h)[012345]\d
)
# 次に同じ形式の任意の秒数を追加
(?::[012345]\d)?
# 最後に任意の am または pm を追加（. とスペースあり可）
(?:\s*(?:a|p)\.?m\.?)?
)"""

COMBINED = f"""(?P<combined>
    (?:
        # 時刻の後に日付、または日付の後に時刻
        {TIME}?{DATE_PATTERN}{TIME}?
        |
        # または上記と同様だが日付の数値バージョン
        {TIME}?{NUM_DATE}{TIME}?
    ) 
    # または単独の時刻
    |
    (?:{TIME})
)"""

price_regex = "(((?:\\d+\.)*\\d+,\\d+)|(\\d+\.\\d+))(?:[/\\s]*)(?:(gbp|\%))"

date_pattern = re.compile(COMBINED, re.IGNORECASE | re.VERBOSE | re.UNICODE)
