package org.apache.spark.ml.made

import org.scalatest.flatspec._
import org.scalatest.matchers._
import org.apache.spark.sql.SparkSession


class StartSparkTest extends AnyFlatSpec with should.Matchers {
  "Spark" should "start context" in {
    val spark = SparkSession.builder
      .master("local[*]")
      .appName("linreg")
      .getOrCreate()

    Thread.sleep(60000)
  }
}
