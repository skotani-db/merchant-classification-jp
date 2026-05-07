<img src=https://raw.githubusercontent.com/databricks-industry-solutions/.github/main/profile/solacc_logo.png width="600px">

[![DBR](https://img.shields.io/badge/DBR-10.4ML-red?logo=databricks&style=for-the-badge)](https://docs.databricks.com/release-notes/runtime/10.4ml.html)
[![CLOUD](https://img.shields.io/badge/CLOUD-ALL-blue?logo=googlecloud&style=for-the-badge)](https://databricks.com/try-databricks)
[![POC](https://img.shields.io/badge/POC-10_days-green?style=for-the-badge)](https://databricks.com/try-databricks)

*[Nilson Report](https://nilsonreport.com/) の2020年調査によると、世界中で毎日約10億件のカード取引が発生しています（米国だけでも1億件）。これは毎日10億件のデータポイントが生まれることを意味し、リテールバンクや決済処理会社にとって、顧客の消費行動をより深く理解し、モバイルバンキングアプリを通じた顧客体験を向上させ、カスタマー360の文脈で大きなクロスセル機会を生み出し、また個別化されたインサイトを活用して不正行為を抑制するための貴重なデータ源となります。カード取引の承認・決済には多くの関係者が関与しており、加盟店からリテールバンクへ伝達される情報は複雑で、時に誤解を招くこともあるため、ブランドや加盟店情報を正確に抽出するには高度な分析手法が必要です。本ソリューションアクセラレーターでは、レイクハウスアーキテクチャがバンク・オープンバンキングアグリゲーター・決済処理会社にとってリテールバンキングの核心的課題である加盟店分類にどのように対応できるかを実証します。*


___
<milos.colic@databricks.com>

___

<img src=https://raw.githubusercontent.com/databricks-industry-solutions/merchant-classification/main/images/reference_architecture.png width="800px">

___

&copy; 2021 Databricks, Inc. All rights reserved. 本ノートブックのソースコードは [Databricks ライセンス](https://databricks.com/db-license-source) に従って提供されています。含まれる、または参照されるすべてのサードパーティライブラリは以下のライセンスに従います。

| ライブラリ                                            | 説明                    | ライセンス | ソース                                              |
|-------------------------------------------------------|-------------------------|------------|-----------------------------------------------------|
| fasttext                                              | NLP ライブラリ          | BSD License| https://fasttext.cc/                                   |
| PyYAML                                 | YAML ファイルの読み込み | MIT        | https://github.com/yaml/pyyaml                      |

## 使い方
このアクセラレーターを実行するには、本リポジトリを Databricks ワークスペースにクローンしてください。Databricks ウェブサイトに公開されているバージョンのノートブックを実行したい場合は `web-sync` ブランチに切り替えてください。DBR 11.0 以降のランタイムを持つ任意のクラスターに `RUNME` ノートブックをアタッチし、「すべて実行」でノートブックを実行してください。アクセラレーターのパイプラインを記述したマルチステップジョブが作成され、そのリンクが表示されます。マルチステップジョブを実行してパイプラインの動作を確認してください。ジョブの設定は RUNME ノートブックに JSON 形式で記述されています。アクセラレーター実行に伴うコストはユーザーの責任となります。
