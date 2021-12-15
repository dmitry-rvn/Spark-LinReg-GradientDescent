package org.apache.spark.ml.made

import com.google.common.io.Files
import org.scalatest.flatspec._
import org.scalatest.matchers._
import breeze.linalg.{*, DenseMatrix, DenseVector}
import org.apache.spark.sql.Row
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{DataFrame, Dataset}


class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {

  val delta: Double = 0.001
  lazy val xyDataset: Dataset[_] = LinearRegressionTest._xyDataset
  lazy val trueWeights: DenseVector[Double] = LinearRegressionTest._trueWeights
  lazy val trueWeightsWithBias: DenseVector[Double] = DenseVector.vertcat(DenseVector(0.0), trueWeights)
  lazy val yVector: DenseVector[Double] = LinearRegressionTest._yVector

  private def validateWeights(weights: DenseVector[Double]): Unit = {
    for (i <- 0 until weights.length) {
      weights(i) should be(trueWeightsWithBias(i) +- delta)
    }
  }

  private def validateTransformedDataframe(transformed: DataFrame): Unit = {
    transformed.columns should be(Seq("x", "y", "y_pred"))
    transformed.collect().length should be(xyDataset.collect().length)

    val predicted: Array[Row] = transformed.select("y_pred").collect()
    predicted.toVector(0).getDouble(0) should be(yVector(0) +- delta)
    for (i <- 0 until 10) {
      predicted.toVector(i).getDouble(0) should be(yVector(i) +- delta)
    }
  }

  "Estimator" should "fit data (with bias)" in {
    val estimator = new LinearRegression().setLearnBias(true)
    val fittedEstimator = estimator.fit(xyDataset)

    validateWeights(fittedEstimator.weights)
  }

  "Estimator" should "fit data (without bias)" in {
    val estimator = new LinearRegression().setLearnBias(false)
    val fittedEstimator = estimator.fit(xyDataset)

    validateWeights(fittedEstimator.weights)
  }

  "Model" should "create prediction given weights" in {
    val model = new LinearRegressionModel(trueWeightsWithBias)
    val transformed = model.transform(xyDataset)

    validateTransformedDataframe(transformed)
  }

  "Estimator" should "work after re-read" in {
    val pipeline = new Pipeline().setStages(Array(new LinearRegression()))
    val tmpFolder = Files.createTempDir()
    pipeline.write.overwrite().save(tmpFolder.getAbsolutePath)
    val reRead = Pipeline.load(tmpFolder.getAbsolutePath)
    val estimator = reRead.fit(xyDataset).stages(0).asInstanceOf[LinearRegressionModel]

    validateWeights(estimator.weights)
  }

  "Model" should "work after re-read" in {
    val pipeline = new Pipeline().setStages(Array(new LinearRegression()))
    val fitted = pipeline.fit(xyDataset)
    val tmpFolder = Files.createTempDir()
    fitted.write.overwrite().save(tmpFolder.getAbsolutePath)
    val reRead = PipelineModel.load(tmpFolder.getAbsolutePath)
    val transformed = reRead.transform(xyDataset)

    validateTransformedDataframe(transformed)
  }

}


object LinearRegressionTest extends WithSpark {
  import sqlc.implicits._

  lazy val _trueWeights: DenseVector[Double] = DenseVector(1.5, 0.3, -0.7)

  lazy val _xMatrix: DenseMatrix[Double] = DenseMatrix.rand[Double](1000, 3)
  lazy val _yVector: DenseVector[Double] = _xMatrix * _trueWeights
  lazy val _xyMatrix: DenseMatrix[Double] = DenseMatrix.horzcat(_xMatrix, _yVector.asDenseMatrix.t)
  lazy val _xyFrame: DataFrame = _xyMatrix(*, ::).iterator
    .map(row => Tuple4(row(0), row(1), row(2), row(3)))
    .toSeq.toDF("x1", "x2", "x3", "y")

  lazy val _assembler: VectorAssembler = new VectorAssembler().setInputCols(Array("x1", "x2", "x3")).setOutputCol("x")
  lazy val _xyDataset: Dataset[_] = _assembler.transform(_xyFrame).select("x", "y")
}
