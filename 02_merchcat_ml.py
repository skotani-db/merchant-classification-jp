# Databricks notebook source
# MAGIC %md
# MAGIC # 加盟店の学習
# MAGIC モデリングはオッカムの剃刀を念頭に置いてシンプルさを優先して始めます。最初のモデルは、前のノートブックで導入したトレーニングデータの5%のみを使用して [`fasttext`](https://fasttext.cc/) アルゴリズムのデフォルトパラメーターのみを使用します。このモデルがベースラインモデルとなり、追加の複雑さはすべて初期モデルのパフォーマンスを向上させるものでなければなりません。

# COMMAND ----------

# MAGIC %run ./config/configure_notebook

# COMMAND ----------

# MAGIC %md
# MAGIC 前のノートブックでは `fasttext` アルゴリズムと互換性のあるトレーニングファイルを生成しました。最新のファイルとバリデーションデータを読み込みます。ファイルは分散ストレージ（例：dbfs:）に保存しましたが、エグゼキューター全体でそのまま読み込むためにはディスクとしてマウントされたストレージの場所が必要です（AWS の場合は[こちら](https://docs.databricks.com/data/data-sources/aws/amazon-s3.html#mount-an-s3-bucket)、Azure の場合は[こちら](https://docs.databricks.com/data/data-sources/azure/azure-storage.html)を参照）。

# COMMAND ----------

display(dbutils.fs.ls(config['model']['train']['hex']))

# COMMAND ----------

# 分散ストレージはマウントされてファイルとしてアクセス可能でなければならない
# ファイルはパーティション1つに結合されているため、トレーニングセット全体をそのまま読み込める
import re
training_file = dbutils.fs.ls(f"{config['model']['train']['hex']}/final")[0].path
training_file = re.sub('dbfs:', '/dbfs', training_file)

# COMMAND ----------

# MAGIC %md
# MAGIC 先ほど生成したバリデーションセットも読み込みます。これはモデルの精度評価に使用します。サンプリング戦略を考慮して、トレーニングセットとテストセットを結合することで加盟店の学習に使用できるレコード数を把握します。

# COMMAND ----------

validation_data = (
  spark
    .read
    .format("delta")
    .load(config['model']['train']['raw'])
    .groupBy('tr_merchant')
    .count()
    .join(spark.read.format("delta").load(config['model']['test']['raw']), ['tr_merchant'], 'left')
    .orderBy('count')
    .withColumnRenamed('count', 'training_records')
)

validation_pdf = validation_data.toPandas()
display(validation_pdf[['tr_description_clean', 'tr_merchant', 'training_records']].sample(100))

# COMMAND ----------

input_features = validation_pdf["tr_description_clean"]
input_targets  = validation_pdf["tr_merchant"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fasttext の構造
# MAGIC ソリューションを実装してシリアライズの課題に取り組む前に、ベースラインモデルを確立し、各パーツを理解しましょう。`fasttext` とそのハイパーパラメーターの詳細については[ドキュメント](https://fasttext.cc/docs/en/python-module.html#train_supervised-parameters)を参照してください。

# COMMAND ----------

import fasttext

model = fasttext.train_supervised(
    input=training_file,
    lr=0.1,
    dim=100,
    ws=5,
    epoch=5,
    minCount=1,
    minCountLabel=1,
    minn=0,
    maxn=0,
    neg=5,
    wordNgrams=5,
    loss="softmax",
    bucket=2000000,
    thread=4,
    lrUpdateRate=100,
    t=0.0001,
    label="__label__",
    verbose=2
)

# COMMAND ----------

# MAGIC %md
# MAGIC 各予測クラスに対するモデルの精度を取得できます。

# COMMAND ----------

import re
result = input_targets.to_frame()
result.columns = ["pr_merchant"]

def predict_label(desc):
  prediction = model.predict(desc)[0][0]
  prediction = re.sub('__label__', '', prediction)
  prediction = re.sub('-', ' ', prediction)
  return prediction

# 正解予測を集計
result["prediction"] = input_features.apply(lambda x: predict_label(x))
result["accuracy"] = result["prediction"] == result["pr_merchant"]
result["accuracy"] = result["accuracy"].apply(lambda x: float(x))
accuracies = result.groupby(["pr_merchant"])["accuracy"].mean()

# 予測された加盟店を表示
df = accuracies.to_frame().sort_values(by='accuracy', ascending=False)
df['pr_merchant'] = accuracies.index
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC 学習対象の加盟店数（1000）を考慮し、異なる分位点の統計を集計します。

# COMMAND ----------

metrics = [
    ["avg__acc", accuracies.mean()],
    ["q_05_acc", accuracies.quantile(0.05)],
    ["q_25_acc", accuracies.quantile(0.25)],
    ["q_50_acc", accuracies.median()],
    ["q_75_acc", accuracies.quantile(0.75)],
    ["q_95_acc", accuracies.quantile(0.95)]
]

import pandas as pd
display(pd.DataFrame(metrics, columns=['metric', 'value']))

# COMMAND ----------

import pyspark.sql.functions as F
import pandas as pd

df = pd.DataFrame(accuracies)
df['pr_merchant'] = df.index
display(spark.createDataFrame(df[['pr_merchant', 'accuracy']]).orderBy(F.desc('accuracy')))

# COMMAND ----------

# MAGIC %md
# MAGIC 一部の加盟店では100%に近い精度ですが、多くのブランドでモデルの精度が大幅に低下し、中央値スコアがほぼゼロに近い結果となっています。これはおそらく加盟店ナレーティブに使用される文字の多様性や加盟店ごとのデータ量の大きな格差によるものです。次のセクションでは、異なるパラメーターや異なるサンプルサイズのトレーニングファイルを試し、十分な精度でより多くの加盟店をカバーできるようにします。

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pyfunc モデル
# MAGIC モデルをさらに調整する前に、パラメーターとメトリクスを追跡できるよう MLflow とモデルトレーニングを統合することで、より高いガバナンスフレームワークの恩恵を受けたいと思います。多くの ML ツールキットと比較して、`fasttext` モデルは cloudpickle 形式でシリアライズできないため、標準的な方法では MLflow で追跡できません。幸い、mlflow にはこの問題を解決するために使用できる `pyfunc` クラスがあります。モデルを mlflow のアーティファクトとしてシリアライズする代わりに「シェルモデルアプローチ」を使用します。パラメーター・メトリクスおよび `/dbfs` のような分散ストレージにモデルを保存した場所を追跡します。`clear_context` メソッドは MLFlow シリアライズ前にメモリ上のモデルを破棄することを確認します。

# COMMAND ----------

from utils.merchcat_utils import *

# COMMAND ----------

# fasttext モデルは cloudpickle で自動的にピクルスできないため、Volume にモデルを保存する
fasttext_home = f"/Volumes/{config['model']['catalog']}/{config['model']['schema']}/fasttext"

# COMMAND ----------

# MAGIC %md
# MAGIC デフォルトのハイパーパラメーターで `fasttext` モデルをブートストラップし、入力データの場所とモデルの出力場所のみを指定します。

# COMMAND ----------

params = {
    "input": training_file,
    "model_location": fasttext_home
}

# COMMAND ----------

from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

with mlflow.start_run(run_name='fasttext-model') as run:
  
  # mlflow ランIDを取得
  run_id = run.info.run_id
  
  # パラメーターを記録
  mlflow.log_params(params)
  
  # モデルを訓練
  fasttextMLF = FastTextMLFlowModel(params, run_id)
  fasttextMLF.train()
  
  # モデルを評価
  metrics = fasttextMLF.evaluate(input_features, input_targets)
  mlflow.log_metrics(metrics)
  
  # シグネチャ付きでモデルを記録
  input_schema = Schema([ColSpec("string", "input")])
  output_schema = Schema([ColSpec("string", "pr_merchant")])
  signature = ModelSignature(inputs=input_schema, outputs=output_schema)
  
  # シリアライズ前にモデルを破棄
  fasttextMLF.clear_context()
  
  # pyfunc モデルをシリアライズ
  mlflow.pyfunc.log_model(
    artifact_path="model", 
    python_model=fasttextMLF, 
    signature=signature
  )

# COMMAND ----------

# MAGIC %md
# MAGIC Python の `fasttextMLF.evaluate` 関数からメトリクスを簡単に抽出できます。パラメーターを変更していないため、当然ながら先ほどと同様のメトリクスが期待されますが、このエンジニアリングアプローチにより複数の MLFlow 実験全体でこれらのメトリクスを追跡し、時間の経過とともにより高い精度を達成できます。

# COMMAND ----------

from mlflow.tracking import MlflowClient
client = MlflowClient()
metrics = client.get_run(run_id).data.metrics
df = pd.DataFrame.from_dict(metrics, orient='index', columns=['value'])
df['metric'] = df.index
display(df[['metric', 'value']])

# COMMAND ----------

# MAGIC %md
# MAGIC トレーニングデータの5%サンプルで上記のメトリクスを持つモデルを得ました。これらのメトリクスから、モデルは（限られたデータにもかかわらず）少なくとも25%の加盟店を適切な精度で学習できたことがわかります。中央値・平均値は望ましいレベルではなく、50%以上の加盟店を全く検出できていません。次のセクションでは、[hyperopts](http://hyperopt.github.io/hyperopt/) を活用してモデルを異なるパラメーターで調整し、パラメーターの変更がパフォーマンスに影響するかどうかを確認します。

# COMMAND ----------

# MAGIC %md
# MAGIC ## ハイパーパラメーターチューニング
# MAGIC いよいよモデルのパフォーマンスについて話し合う準備ができました。異なるパラメーター（`epoch` 数や `ngrams` など）で手動でモデルを再訓練することもできますが、最小限のオーバーヘッドで `hyperopt` にその網羅的な作業を委ねることができます。[Hyperopt](https://docs.databricks.com/applications/machine-learning/automl-hyperparam-tuning/index.html#hyperparameter-tuning-with-hyperopt) は spark の上でハイパーパラメーターチューニングを実行できるフレームワークです。

# COMMAND ----------

from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK, Trials, pyll
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

def train_and_log_fasttext(run, run_id, params):
  
  fasttext_params = {
      "input": params['training_file'],
      "model_location": fasttext_home,
      "lr": params['lr'],
      "epoch": int(params['epochs']),
      "wordNgrams": int(params['ngram_size']),
      "dim": int(params['dimensions'])
  }
    
  # モデルを作成
  fasttextMLF = FastTextMLFlowModel(fasttext_params, run_id)
  
  # パラメーターを記録
  mlflow.log_params(fasttext_params)
  
  # サンプルサイズをファイル名として保存
  mlflow.log_param("sample-size", params['training_file'].split('.txt')[0].split('-')[-1])
  
  # モデルを訓練
  fasttextMLF.train()
  
  # メトリクスを評価
  metrics = fasttextMLF.evaluate(input_features, input_targets)
  mlflow.log_metrics(metrics)
  
  # シグネチャ付きでモデルを記録
  input_schema = Schema([ColSpec("string", "input")])
  output_schema = Schema([ColSpec("string", "pr_merchant")])
  signature = ModelSignature(inputs=input_schema, outputs=output_schema)
  
  # シリアライズ前にモデルを破棄
  fasttextMLF.clear_context()
  
  # pyfunc モデルをシリアライズ
  mlflow.pyfunc.log_model(
    artifact_path="model", 
    python_model=fasttextMLF, 
    signature=signature
  )

  # 損失関数を返す
  loss = -metrics['avg__acc']
  return {'loss': loss, 'status': STATUS_OK, 'params': fasttext_params, 'run_id': run_id}

# COMMAND ----------

def hyper_train_model(params):
  with mlflow.start_run(run_name='fasttext-model', nested=True) as run:
    run_id = run.info.run_id
    run_result = train_and_log_fasttext(run, run_id, params)
    return run_result

# COMMAND ----------

# MAGIC %md
# MAGIC `hyperopt` と spark で複数のモデルをトレーニングするには、探索空間と spark trials を定義する必要があります。この目的のために、6次元にわたる複雑な探索空間を定義して、一度に X 個のモデル（合計25モデル）を訓練します。`fasttext` を含む多くのアルゴリズムは、特定のマシン上で複数のスレッドを活用できます。一方、Spark は各タスクが1スレッドのみを必要とすると仮定しています。Spark のデフォルト設定のままでは、単一ノードで単一モデルを実行することになり、個々のモデルの実行時間が大幅に遅くなります。代わりに、ハイパーパラメーターチューニングタスク専用のクラスターを用意します。8コアを持つ5ノードのクラスターを作成し、`spark.task.cpus` も8に設定します。これにより `hyperopt` と spark は各ワーカーノードで正確に1つのモデルを実行することができます。

# COMMAND ----------

search_space = {
  'training_file': training_file,
  'lr': hp.uniform('lr', 0.05, 0.4),
  'epochs': hp.quniform('epochs', 5, 15, 1),
  'ngram_size': hp.quniform('ngram_size', 2, 4, 1),
  'dimensions': hp.quniform('dimensions', 20, 120, 10)
}

# COMMAND ----------

spark_trials = SparkTrials(parallelism=config['model']['executors'], spark_session=spark)

argmin = fmin(
  fn=hyper_train_model,
  space=search_space,
  algo=tpe.suggest,
  max_evals=25,
  trials=spark_trials
)

# COMMAND ----------

# MAGIC %md
# MAGIC `MlflowClient` と `spark_trials` を組み合わせることで、最良のモデルのモデル精度をプログラムで取得できます。以下に示すように、少なくとも25%のレコードで完全な精度を達成し、最悪の5%のイベントでも90%の精度を維持しています。全体として、数千ブランド・数百万件のカード取引に対して97%の確率で加盟店名を正確に予測することができました。

# COMMAND ----------

from mlflow.tracking import MlflowClient

best_run_id = spark_trials.best_trial['result']['run_id']
client = mlflow.tracking.MlflowClient()

best_metrics = client.get_run(best_run_id).data.metrics
best_metrics.pop('loss')

df = pd.DataFrame.from_dict(best_metrics, orient='index', columns=['value'])
df['metric'] = df.index
display(df[['metric', 'value']])

# COMMAND ----------

# MAGIC %md
# MAGIC `hyperopts` の `space_eval` 関数を使用して最適なパラメーターにもアクセスでき、専門家の意見ではなく実証的な結果に基づいて最良のモデルを取得できます。以下に示すように、上記のメトリクスを示した最良の実験は `epoch` 数14、`ngram_size` 3を使用して実施されました。

# COMMAND ----------

from hyperopt import space_eval
best_model_params = space_eval(search_space, argmin)
df = pd.DataFrame.from_dict(best_model_params, orient='index', columns=['value'])
df = df.astype(str)
df['param'] = df.index
display(df[['param', 'value']])

# COMMAND ----------

# MAGIC %md
# MAGIC 最後に、MLFlow ユーザーインターフェースを使用してすべての実験を並べて比較し、各パラメーターが全体的なモデル精度に与える影響をより深く理解できます。

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://raw.githubusercontent.com/databricks-industry-solutions/merchant-classification/main/images/merchcat_hyperopts_1.png" width="800px">

# COMMAND ----------

# MAGIC %md
# MAGIC MLFlow と `pyfunc` を使用することで、数百万件のカード取引ナレーティブに隠れた数千の加盟店名を正しく分類できるモデルを訓練することができました。ただし、このアプローチは学習対象のクリーンな加盟店名がすでに存在するという前提に基づいています。実際のブランド情報で数千件のカード取引にラベルを付けて開始しましたが、そのような作業に必要な労力を認識しています。必要なラベルのサイズと品質は、次のセクションで実際の実証結果をもとに評価します。

# COMMAND ----------

# MAGIC %md
# MAGIC ## 最小限のトレーニングデータは？
# MAGIC 前述のとおり、初期データの5%サンプルのみを使用しました。より多くのカード取引にラベルを付けることでモデルが大幅に改善されるかどうかが気になります。数千件のトレーニングデータを追加することは労力に見合う価値があるのでしょうか？この目的のために、再度 `hyperopt` と spark を活用します。ユーティリティノートブック（`%run` コマンドとしてインポート）を使用して、トレーニングデータの5%から30%の範囲でさまざまなサブサンプルを生成します。これらのファイルの場所をモデル最適化戦略のもう一つのハイパーパラメーターとして設定します。

# COMMAND ----------

# MAGIC %run ./utils/fasttext_utils

# COMMAND ----------

tf = TrainingFile(
    dataframe_location=config['model']['train']['raw'],
    output_location=config['model']['train']['hex'],
    target_column='tr_merchant',
    fasttext_column='fasttext'
)

file_thresholds = [0.3, 0.25, 0.2, 0.15, 0.10, 0.05]
training_files = [tf.generate_training_file(sample_rate=t, min_count=50) for t in file_thresholds]
training_files = [f'/dbfs{training_file}' for training_file in training_files]

# COMMAND ----------

# MAGIC %md
# MAGIC この演習では90モデルを並列でトレーニングします。`hp.choice` 関数を使用して、異なる実験に異なるトレーニングファイルを提供するために可能なサンプル場所のコレクションから1つのオプションを選択できるようにします。

# COMMAND ----------

search_space = {
  'training_file': hp.choice('training_file', training_files),
  'lr': hp.uniform('lr', 0.05, 0.4),
  'epochs': hp.quniform('epochs', 5, 15, 1),
  'ngram_size': hp.quniform('ngram_size', 2, 4, 1),
  'dimensions': hp.quniform('dimensions', 20, 120, 10)
}

spark_trials = SparkTrials(parallelism=config['model']['executors'], spark_session=spark)
  
argmin = fmin(
  fn=hyper_train_model,
  space=search_space,
  algo=tpe.suggest,
  max_evals=90,
  trials=spark_trials
)

# COMMAND ----------

# MAGIC %md
# MAGIC 今度は、トレーニングサンプルサイズを定義する新しい入力パラメーターを加えた実験を並べて比較できます。

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://raw.githubusercontent.com/databricks-industry-solutions/merchant-classification/main/images/merchcat_hyperopts_2.png" width="800px">

# COMMAND ----------

# MAGIC %md
# MAGIC 探索の結果は非常に有益でした。**初期トレーニングデータのわずか30%で望ましい予測パフォーマンスを維持できること**が証明されました。末尾の加盟店のレコード数を確認すると、ラベル付きデータが44件しかない加盟店が実際に存在することがわかります。これは、優れたパフォーマンスを維持しながらもさらに学習データを削減できる可能性を示唆しています。

# COMMAND ----------

from mlflow.tracking import MlflowClient

best_run_id = spark_trials.best_trial['result']['run_id']
client = mlflow.tracking.MlflowClient()

best_metrics = client.get_run(best_run_id).data.metrics
best_metrics.pop('loss')

df = pd.DataFrame.from_dict(best_metrics, orient='index', columns=['value'])
df['metric'] = df.index
display(df[['metric', 'value']])

# COMMAND ----------

from hyperopt import space_eval
best_model_params = space_eval(search_space, argmin)
df = pd.DataFrame.from_dict(best_model_params, orient='index', columns=['value'])
df = df.astype(str)
df['param'] = df.index
display(df[['param', 'value']])

# COMMAND ----------

# MAGIC %md
# MAGIC ## モデル推論
# MAGIC 元の入力取引セットから加盟店を推論する前に、最良の実験を MLRegistry のモデル候補として登録しましょう。実際のシナリオではモデルのレビューが必要ですが、ここではプログラムで本番アーティファクトとして利用可能にします。組織は MLFlow に webhook を作成することで、モデルを上位の環境にプロモートする前にレビューが必要な新しいモデルの通知を独立検証部門（IVU プロセス）に送ることができます。

# COMMAND ----------

model_uri = f'runs:/{best_run_id}/model'
result = mlflow.register_model(model_uri, config['model']['name'])
version = result.version

# COMMAND ----------

client = mlflow.tracking.MlflowClient()
client.set_registered_model_alias(
    name=config['model']['name'],
    alias="champion",
    version=version
)

# COMMAND ----------

logged_model = f"models:/{config['model']['name']}@champion"
loaded_model = mlflow.pyfunc.load_model(logged_model)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pandas として読み込む
# MAGIC pandas データフレームに収まる小規模なサンプルをスコアリングする場合は `model.predict` メソッドを使用します。

# COMMAND ----------

test_pdf = validation_pdf.head(5000).sample(1000)
test_pdf["input"] = test_pdf["tr_description_clean"]
test_pdf["pr_merchant"] = loaded_model.predict(test_pdf)
display(test_pdf[["tr_description", "tr_merchant", "pr_merchant"]])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Spark として読み込む
# MAGIC spark で利用可能なより大きなデータフレームをスコアリングする場合、mlflow によって自動生成された spark `udf` を使用します。

# COMMAND ----------

merchant = mlflow.pyfunc.spark_udf(
  spark, 
  model_uri=logged_model, 
  result_type="string"
)

spark_results = validation_data.withColumn('pr_merchant', merchant("tr_description_clean"))
display(spark_results.select("tr_description", "tr_merchant", "pr_merchant"))

# COMMAND ----------

# MAGIC %md
# MAGIC これら2つのアプローチにより、マイクロバッチと数億件以上の取引を処理する必要がある大規模な過去ジョブの両方に対応できます。モデルラッパークラスで提供される spark `udf` API の利点は、カード取引をリアルタイムでブランド情報で充実させることができる構造化ストリーミングアプローチも解放します。また、実際のシナリオでモデルがどのように動作するかを確認し、正確な予測数を集計します。

# COMMAND ----------

display(
  spark_results
    .withColumn("predicted", F.when(F.col("pr_merchant") == F.col("tr_merchant"), F.lit(1)).otherwise(F.lit(0)))
    .groupBy("tr_merchant")
    .agg(F.sum(F.col("predicted")).alias("predicted"))
    .join(spark_results.groupBy("tr_merchant").count(), ["tr_merchant"])
    .withColumn("accuracy", F.col("predicted") / F.col("count"))
    .orderBy(F.desc("accuracy"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC 上記に示すように、最初のイテレーション以降モデルのカバレッジが大幅に改善されました。新しいカード取引が発生するにつれ、カバレッジと精度のバランスを継続的に監視する必要があります。理想的には、ここで定義されたフレームワークを通じて、出力の品質が基準を満たさなくなった時点で、組織が新しいパターンや新しいラベル（モバイルバンキングアプリ等でのエンドユーザーによる設定など）から自動的に（または最小限の監視で）学習できるようになります。

# COMMAND ----------

# MAGIC %md
# MAGIC ## まとめ
# MAGIC 本ソリューションでは、加盟店分類の問題を短いドキュメントの分類問題として取り組みました。このタスクのモデルとして [`fasttext`](https://fasttext.cc/) を選択し、MLFlow と `hyperopt` との統合に成功しました。この演習を通じて、組織が**加盟店ごとにわずか50件のラベル付きレコード**（5と0だけ！）から高品質な加盟店分類を導入できることを実証しました。この事実は大きな価値を解放します。

# COMMAND ----------

# MAGIC %md
# MAGIC アナリストチームは、この自動化されたソリューションが引き継ぐ前に、初期の真実のソースにラベルを付けるのに数日しかかかりません。その後、アナリストは「オートパイロット」モードに切り替え、不正行為や顧客の消費パターンなどの取引データから抽出できる付加価値に集中できます。堅牢なトランザクション充実化が整備されることで、加盟店が適切に識別されたモバイルバンキング上の取引を提示し、提示内容が正確であるという確信を持って（エンドカスタマーとの高品質と信頼を維持しながら）上記の例のように表示できます。次のソリューションアクセラレーターでは、この分類をビルディングブロックとして活用し、個別化されたインサイトや行動的な取引パターン（トランザクション埋め込み）を推進します。
