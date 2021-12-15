package org.apache.spark.ml.made

import org.apache.spark.sql.{SparkSession, SQLContext}


trait WithSpark {
  lazy val spark: SparkSession = WithSpark._spark
  lazy val sqlc: SQLContext = WithSpark._sqlc
}


object WithSpark {
  lazy val _spark: SparkSession = SparkSession.builder
    .master("local[*]")
    .appName("linreg")
    .getOrCreate()

  lazy val _sqlc: SQLContext = _spark.sqlContext
}
