# Databricks notebook source
# MAGIC %md
# MAGIC <img src=https://d1r5llqwmkrl74.cloudfront.net/notebooks/fs-lakehouse-logo.png width="600px">
# MAGIC 
# MAGIC [![DBR](https://img.shields.io/badge/DBR-10.4ML-red?logo=databricks&style=for-the-badge)](https://docs.databricks.com/release-notes/runtime/10.4ml.html)
# MAGIC [![CLOUD](https://img.shields.io/badge/CLOUD-ALL-blue?logo=googlecloud&style=for-the-badge)](https://databricks.com/try-databricks)
# MAGIC [![POC](https://img.shields.io/badge/POC-10_days-green?style=for-the-badge)](https://databricks.com/try-databricks)
# MAGIC 
# MAGIC *[Nilson Report](https://nilsonreport.com/) の2020年調査によると、世界中で毎日約10億件のカード取引が発生しています（米国だけでも1億件）。これは毎日10億件のデータポイントが生まれることを意味し、リテールバンクや決済処理会社にとって、顧客の消費行動をより深く理解し、モバイルバンキングアプリを通じた顧客体験を向上させ、カスタマー360の文脈で大きなクロスセル機会を生み出し、また個別化されたインサイトを活用して不正行為を抑制するための貴重なデータ源となります。カード取引の承認・決済には多くの関係者が関与しており、加盟店からリテールバンクへ伝達される情報は複雑で、時に誤解を招くこともあるため、ブランドや加盟店情報を正確に抽出するには高度な分析手法が必要です。本ソリューションアクセラレーターでは、レイクハウスアーキテクチャがバンク・オープンバンキングアグリゲーター・決済処理会社にとってリテールバンキングの核心的課題である加盟店分類にどのように対応できるかを実証します。*
# MAGIC 
# MAGIC 
# MAGIC ___
# MAGIC <milos.colic@databricks.com>

# COMMAND ----------

# MAGIC %md
# MAGIC <img src=https://raw.githubusercontent.com/databricks-industry-solutions/merchant-classification/main/images/reference_architecture.png width="800px">

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved. 本ノートブックのソースコードは [Databricks ライセンス](https://databricks.com/db-license-source) に従って提供されています。含まれる、または参照されるすべてのサードパーティライブラリは以下のライセンスに従います。
# MAGIC 
# MAGIC | ライブラリ                                            | 説明                    | ライセンス | ソース                                              |
# MAGIC |-------------------------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | fasttext                                              | NLP ライブラリ          | BSD License| https://fasttext.cc/                                   |
# MAGIC | PyYAML                                 | YAML ファイルの読み込み | MIT        | https://github.com/yaml/pyyaml                      |
