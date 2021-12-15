package org.apache.spark.ml.made

import breeze.linalg.{DenseVector, sum}
import breeze.linalg.functions.euclideanDistance
import org.apache.spark.sql.{DataFrame, Dataset, Encoder}
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.types.{DoubleType, StructType}
import org.apache.spark.sql.functions.lit
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.{BooleanParam, DoubleParam, LongParam, Param, ParamMap}
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsReader, DefaultParamsWritable,
  DefaultParamsWriter, Identifiable, MLReadable, MLReader, MLWritable, MLWriter, SchemaUtils}
import org.apache.spark.mllib
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer


trait LinearRegressionParams extends HasInputCol with HasOutputCol {
  def setInputCol(value: String) : this.type = set(inputCol, value)
  setDefault(inputCol -> "x")

  def setOutputCol(value: String) : this.type = set(outputCol, value)
  setDefault(outputCol -> "y")

  val predictionCol: Param[String] = new Param[String](this, "predictionCol", "prediction column name")
  def getPredictionCol : String = $(predictionCol)
  def setPredictionCol(value: String): this.type = set(predictionCol, value)
  setDefault(predictionCol -> "y_pred")

  val learningRate = new DoubleParam(this, "learningRate", "learning rate")
  def getLearningRate : Double = $(learningRate)
  def setLearningRate(value: Double) : this.type = set(learningRate, value)
  setDefault(learningRate -> 0.05)

  val maxIteration = new LongParam(this, "maxIteration", "max number of iterations")
  def getMaxIteration : Long = $(maxIteration)
  def setMaxIteration(value: Long) : this.type = set(maxIteration, value)
  setDefault(maxIteration -> 10000)

  val tolerance = new DoubleParam(this, "tolerance", "tolerance for weights change to converge")
  def getTolerance : Double = $(tolerance)
  def setTolerance(value: Double) : this.type = set(tolerance, value)
  setDefault(tolerance -> 0.000001)

  val learnBias = new BooleanParam(this, "learn bias", "whether to learn bias weight")
  def isLearnBias : Boolean = $(learnBias)
  def setLearnBias(value: Boolean) : this.type = set(learnBias, value)
  setDefault(learnBias -> false)

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getInputCol, new VectorUDT())
    if (schema.fieldNames.contains(getOutputCol)) {
      SchemaUtils.checkColumnType(schema, getOutputCol, DoubleType)
      schema
    } else {
      SchemaUtils.appendColumn(schema, schema(getOutputCol).copy(name = getOutputCol))
    }
  }
}


class LinearRegression(override val uid: String)
  extends Estimator[LinearRegressionModel] with LinearRegressionParams with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("linearRegression"))

  override def fit(dataset: Dataset[_]): LinearRegressionModel = {
    implicit val encoder: Encoder[Vector] = ExpressionEncoder()

    val datasetWithBias = dataset.withColumn("b", lit(if (isLearnBias) 1.0 else 0.0))
    val assembler = new VectorAssembler().setInputCols(Array("b", "x", "y")).setOutputCol("bxy")
    val bxyVectors: Dataset[Vector] = assembler.transform(datasetWithBias).select("bxy").as[Vector]

    val wSize: Int = bxyVectors.first().size - 1

    var prevWeights: DenseVector[Double] = DenseVector.fill(wSize){Double.PositiveInfinity}
    val weights: DenseVector[Double] = DenseVector.fill(wSize){0.0}

    var iteration: Long = 0
    while (iteration < getMaxIteration && euclideanDistance(weights.toDenseVector, prevWeights.toDenseVector) > getTolerance) {

      val summary = bxyVectors.rdd.mapPartitions((data: Iterator[Vector]) => {
        val summarizer = new MultivariateOnlineSummarizer()
        data.foreach(row => {
          // row = (bias, x1, x2, ..., xn, y)
          val x = row.asBreeze(0 until wSize).toDenseVector
          val yTrue = row.asBreeze(-1)
          val yPred = x.dot(weights)
          val residuals = yTrue - yPred
          val weightsDelta = -2.0 * (x * residuals)
          summarizer.add(mllib.linalg.Vectors.fromBreeze(weightsDelta))
        })
        Iterator(summarizer)
      }).reduce(_ merge _)

      prevWeights = weights.copy
      weights -= getLearningRate * summary.mean.asBreeze

      iteration += 1
    }

    copyValues(new LinearRegressionModel(weights).setParent(this))
  }

  override def copy(extra: ParamMap): LinearRegression = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)
}


object LinearRegression extends DefaultParamsReadable[LinearRegression]


class LinearRegressionModel(override val uid: String, val weights: DenseVector[Double])
  extends Model[LinearRegressionModel] with LinearRegressionParams with MLWritable {

  def this(weights: DenseVector[Double]) = this(Identifiable.randomUID("linearRegressionModel"), weights)

  override def transform(dataset: Dataset[_]): DataFrame = {
    val transformUdf = {
      dataset.sqlContext.udf.register(uid + "_transform",
        (x : Vector) => {
          // weights = (bias, x1, x2, ..., xn)
          weights(0) + sum(x.asBreeze.toDenseVector * weights(1 until weights.length))
        }
      )
    }
    dataset.withColumn(getPredictionCol, transformUdf(dataset(getInputCol)))
  }

  override def copy(extra: ParamMap): LinearRegressionModel = copyValues(new LinearRegressionModel(weights), extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  override def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)
      val vectors = Tuple1(Vectors.fromBreeze(weights))
      sqlContext.createDataFrame(Seq(vectors)).write.parquet(path + "/vectors")
    }
  }
}


object LinearRegressionModel extends MLReadable[LinearRegressionModel] {
  override def read: MLReader[LinearRegressionModel] = new MLReader[LinearRegressionModel] {
    override def load(path: String): LinearRegressionModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc)
      val vectors = sqlContext.read.parquet(path + "/vectors")
      implicit val encoder : Encoder[Vector] = ExpressionEncoder()
      val weights = vectors.select(vectors("_1").as[Vector]).first().asBreeze.toDenseVector
      val model = new LinearRegressionModel(weights)
      metadata.getAndSetParams(model)
      model
    }
  }
}
