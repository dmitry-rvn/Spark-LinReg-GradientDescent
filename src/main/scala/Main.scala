package org.apache.spark.ml.made

import breeze.linalg.{*, DenseMatrix, DenseVector}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession


object Main{
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .master("local[*]")
      .appName("linreg")
      .getOrCreate()

    import spark.implicits._

    val x = DenseMatrix.rand[Double](100000, 3)
    val w = DenseVector(1.5, 0.3, -0.7)
    val y = x * w

    val xy = DenseMatrix.horzcat(x, y.asDenseMatrix.t)
    val xyFrame = xy(*, ::).iterator
      .map(row => Tuple4(row(0), row(1), row(2), row(3)))
      .toSeq.toDF("x1", "x2", "x3", "y")
    val assembler = new VectorAssembler().setInputCols(Array("x1", "x2", "x3")).setOutputCol("x")
    val xyDataset = assembler.transform(xyFrame).select("x", "y")

    val regressor = new LinearRegression()
    val fittedRegressor = regressor.fit(xyDataset)

    println("True weights (bias = 0):")
    println(w)
    println("Regressor parameters:")
    println(s"- Learning rate: ${regressor.getLearningRate}")
    println(s"- Learn bias: ${regressor.isLearnBias}")
    println(s"- Max iteration: ${regressor.getMaxIteration}")
    println(s"- Tolerance: ${regressor.getTolerance}")
    println("Learned weights (bias, x1, x2, ..., xn):")
    println(fittedRegressor.weights)

    fittedRegressor.transform(xyDataset).show(5)
  }
}
