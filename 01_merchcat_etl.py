# Databricks notebook source
# MAGIC %md
# MAGIC # 前処理
# MAGIC 
# MAGIC 取引ナレーティブと加盟店説明は、共通のガイドラインや業界標準なしに加盟店が自由に入力するフリーテキストであるため、このデータの不整合問題にはデータサイエンス的なアプローチが必要です。本ソリューションアクセラレーターでは、テキスト分類技術を活用して、加盟店の参照データセットをもとに任意の取引ナレーティブに隠されたブランドをよりよく理解する方法を実証します。`STARBUCKS LONDON 1233-242-43 2021` という取引説明は、企業名「Starbucks」にどれだけ近いでしょうか？

# COMMAND ----------

# MAGIC %run ./config/configure_notebook

# COMMAND ----------

# MAGIC %md
# MAGIC システムの出力品質は、機械学習モデルに供給できるデータの品質と直接相関していることは言うまでもありません。適切に構造化された高品質なトレーニングデータとテストデータのサンプルを確保することは、優れた機械学習モデルを訓練することと同様に重要です。まず、販売時点で発生しうる加盟店ナレーティブを含む生の取引データのサンプルから始めます。POS が生成するであろう最も基本的なフォーマットは（日付、金額、説明、カード番号）です。

# COMMAND ----------

from pyspark.sql import functions as F

tr_df = (
    spark
        .read
        .format('delta')
        .load(config['transactions']['raw'])
        .select('tr_date', 'tr_merchant', 'tr_description', 'tr_amount')
        .filter(F.expr('tr_merchant IS NOT NULL'))
)

display(tr_df.select("tr_date", "tr_description", "tr_amount"))

# COMMAND ----------

# MAGIC %md
# MAGIC 本ソリューションアクセラレーターでは、テキスト分類と表現学習のための効率的なフレームワークである [`fasttext`](https://fasttext.cc/) ライブラリの使用方法を実証します。このノートブックの目的は、生のカード取引ナレーティブを `fasttext` モデルに入力できる形式のデータに変換することです。この演習のために、実際のブランド名・加盟店名で数千件のカード取引（`tr_merchant` 列）にラベルを付け、一連のノートブックを通じてさらに精緻化することで、数百万件のラベル付き取引データセットを作成しました。実際には、金融機関のほとんどはすでに加盟店を学習するためのラベルシリーズを保有しています。必要なラベルのサイズと品質は、次のノートブックで実際の実証結果をもとに評価します。

# COMMAND ----------

# MAGIC %md
# MAGIC ## 加盟店ナレーティブ
# MAGIC まず気づくことは、カード取引ナレーティブが非常に非構造化されているという点です。これらの説明はグローバルなフォーマットに従っておらず、しばしば部分的にマスクされたデータを含みます。多くの場合、日付・金額・固有識別子など、カード取引に関連する加盟店を理解する上で有益な情報をもたらさないトークンが含まれています。これを踏まえ、前処理ステップの一環としてデータクレンジングを実施しました。文字列データから日付を除去するための kaggle [記事](https://www.kaggle.com/edrushton/removing-dates-data-cleaning) をもとに、説明文から日付や説明上の価値を持たない不要な文字を除去するための一連のシンプルな正規表現を作成しました。

# COMMAND ----------

from utils.regex_utils import *

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql import types as T
import pandas as pd

@F.pandas_udf("string")
def dates_udf(col: pd.Series) -> pd.Series:
  return col.apply(lambda description: str(date_pattern.sub(" ", str(description))))

tr_df_cleaned = (
    tr_df
        .withColumn("tr_description_clean", dates_udf(F.col("tr_description")))
        .withColumn("tr_description_clean", F.regexp_replace(F.col("tr_description_clean"), price_regex, ""))
        .withColumn("tr_description_clean", F.regexp_replace(F.col("tr_description_clean"), "(\(+)|(\)+)", ""))
        .withColumn("tr_description_clean", F.regexp_replace(F.col("tr_description_clean"), "&", " and "))
        .withColumn("tr_description_clean", F.regexp_replace(F.col("tr_description_clean"), "[^a-zA-Z0-9]+", " "))
        .withColumn("tr_description_clean", F.regexp_replace(F.col("tr_description_clean"), "\\s+", " "))
        .withColumn("tr_description_clean", F.regexp_replace(F.col("tr_description_clean"), "\\s+x{2,}\\s+", " ")) 
        .withColumn("tr_description_clean", F.trim(F.col("tr_description_clean")))
)

display(tr_df_cleaned.select("tr_merchant", "tr_description", "tr_description_clean"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### fasttext フォーマット
# MAGIC クレンジングとサンプリングの演習の一環として、`fasttext` モデルに準拠した形式にデータを整形します。fasttext モデルは特定のフォーマットのデータを必要とします。学習対象のラベルは実際の加盟店名（`tr_merchant`）であり、パターンはクレンジング済みの説明（`tr_description_clean`）です。
# MAGIC 
# MAGIC ```
# MAGIC __label__merchant1 clean description from narrative 1
# MAGIC __label__merchant2 clean description from narrative 2
# MAGIC __label__merchant3 clean description from narrative 3
# MAGIC ```

# COMMAND ----------

tr_df_fasttext = tr_df_cleaned.withColumn(
    "fasttext",
    F.concat(
        F.concat(
            F.lit("__label__"),
            F.regexp_replace(F.col("tr_merchant"), "\\s+", "-")
        ),
        F.lit(" "),
        F.col("tr_description_clean")
    )
)

display(tr_df_fasttext.select("fasttext"))

# COMMAND ----------

# MAGIC %md
# MAGIC この入力データセットを機械学習で使用できるデルタテーブルとして保存します。

# COMMAND ----------

_ = (
    tr_df_fasttext
      .write
      .mode("overwrite")
      .format("delta")
      .save(config['transactions']['fmt'])
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 不均衡データセット
# MAGIC カード取引データでは、加盟店ごとに利用可能なデータの大きな偏りが生じることが非常に一般的です。例えば「Amazon」は「MyLittleCornerShop」よりもはるかに多くの取引を生み出すことが予想されます。生データの分布を確認してみましょう。

# COMMAND ----------

tr_df = spark.read.format('delta').load(config['transactions']['fmt'])
df = tr_df.groupBy("tr_merchant").count().orderBy("count").toPandas()
df.plot.hist(bins=100)

# COMMAND ----------

# MAGIC %md
# MAGIC 「Tesco」と「MyLittleCornerShop」を比較すると、機械学習に利用可能なデータが大きく異なることがわかります。Tesco には学習用のカード取引が数百万件あるのに対し、小規模な店舗では数千件程度にすぎません。では、新しく到着した取引を正常にスコアリングするためにテキストパターンを学習するには、加盟店ごとにどれだけのデータが必要なのでしょうか？この問いに適切に答える唯一の方法はメトリクスと組み合わせることです。各加盟店全体と集団全体に対するパフォーマンスを測定できる必要があります。

# COMMAND ----------

# MAGIC %md
# MAGIC ### サンプリング戦略
# MAGIC 規模が大幅に異なるデータサンプルを提供すると、モデルはより大きなサンプルからしか十分に学習できず、データが少ない加盟店は誤分類として扱われる可能性があります。この問題に対処するために**層化抽出**を使用します。全加盟店のサンプルが最低100件、最大5000件になるようにサンプリングします。

# COMMAND ----------

def format_dict(label_column, value_column, in_dict):
    labels = in_dict[label_column]
    rates = in_dict[value_column]
    result = dict()
    for i in range(0, len(labels)):
        result[labels[i]] = rates[i]
    return result

def sample_data(sample_size, count_threshold, data):
    counted = data.groupBy("tr_merchant").count()
    counted = counted.where(F.col("count") >= count_threshold)
    counted = counted \
        .withColumn("sample_rate", sample_size / F.col("count")) \
        .withColumn("sample_rate", F.when(F.col("sample_rate") > 1, 1).otherwise(F.col("sample_rate")))
    sample_rates = counted.select("tr_merchant", "sample_rate").toPandas().to_dict()
    sample_rates = format_dict("tr_merchant", "sample_rate", sample_rates)
    result = data.sampleBy("tr_merchant", sample_rates)
    return result

# COMMAND ----------

tr_df_sampled = sample_data(5000, 100, tr_df)
df_sampled = tr_df_sampled.groupBy("tr_merchant").count().orderBy("count").toPandas()
df_sampled.plot.hist(bins=100)

# COMMAND ----------

# MAGIC %md
# MAGIC 平均的に利用可能なデータと比べて過少代表のブランドが存在することに気づきますが、これらもそのまま残し、モデルが学習を試みられるようにします。主な目的のひとつは、モデルがブランドを認識するために必要なラベル付けの最小限の努力を金融機関に示すことだからです。この動機は、ラベル付きデータを持たない金融機関に対して**機械学習モデルを訓練するために必要な真実のソースを形成するための最小限の努力**を実証したいという願望に根ざしています（私たち自身がその初期データセットを作成するためにそうしたように）。

# COMMAND ----------

# MAGIC %md
# MAGIC ### トレーニングセット
# MAGIC 機械学習の世界に踏み込む前に、データをトレーニング/バリデーションサンプルに分割する必要があります。これを実現する方法のひとつは、ランダムな順序でデータセットの各行にクラスごとのパーセンタイルを付与することです。これにより、再現可能な方法でデータの10%をバリデーションデータとして抽出できます。

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.window import Window

w =  Window.partitionBy("tr_merchant").orderBy(F.rand())
df = tr_df_sampled.withColumn("class_percentile", F.bround(F.percent_rank().over(w), 4))

# COMMAND ----------

# MAGIC %md
# MAGIC データを分割し、トレーニングセットとバリデーションセットの両方を Delta Lake テーブルに保存します。

# COMMAND ----------

df.where("class_percentile < 0.9") \
  .write \
  .mode("overwrite") \
  .format("delta") \
  .save(config['model']['train']['raw'])

# COMMAND ----------

df.where("class_percentile >= 0.9") \
  .write \
  .mode("overwrite") \
  .format("delta") \
  .save(config['model']['test']['raw'])

# COMMAND ----------

# MAGIC %md
# MAGIC ## fasttext ファイル
# MAGIC 特定のフォーマットに加えて、`fasttext` モデルは単一のテキストファイルからデータを読み込むことを期待します。このファイルの各行は、先ほど強制したフォーマットでなければなりません。ユーティリティノートブックの `TrainingFile` クラスは、spark データフレームを `fasttext` のトレーニングロジックが期待する単一のフラットファイルに変換するために必要なロジックを管理します。全エグゼキューター（ディスクにマウントされた状態）からアクセス可能な指定の出力場所に、固有の名前でファイルを生成します。

# COMMAND ----------

# MAGIC %run ./utils/fasttext_utils

# COMMAND ----------

tf = TrainingFile(
    dataframe_location=config['model']['train']['raw'],
    output_location=config['model']['train']['hex'],
    target_column='tr_merchant',
    fasttext_column='fasttext'
)

# COMMAND ----------

# MAGIC %md
# MAGIC 次のようにしてトレーニングファイルを生成できます。次のノートブックでの初期モデルの結果によっては、異なるサイズのサンプルを生成する必要があるかもしれません。異なるサンプルサイズで生成された各トレーニングファイルには特定のバージョン（UUID）が付与され、MLFlow 実験全体で追跡できます。

# COMMAND ----------

training_file = tf.generate_training_file(
    sample_rate=0.05, 
    min_count=50
)

# COMMAND ----------

display(dbutils.fs.ls(config['model']['train']['hex']))

# COMMAND ----------

input_dir = f"{config['model']['train']['hex']}/final"
display(spark.read.format('text').load(input_dir))

# COMMAND ----------

# MAGIC %md
# MAGIC ### まとめ
# MAGIC これらの最初のセクションでは、データのクレンジングと標準化に相当な労力を費やしました。その動機はシンプルです。高品質なデータほど高品質な機械学習成果をもたらします。`fasttext` のトレーニングファイルが整ったところで、カード取引ナレーティブから加盟店を抽出するための初期モデルを訓練できます。
