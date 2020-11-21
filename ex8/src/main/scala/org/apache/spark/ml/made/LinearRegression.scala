package org.apache.spark.ml.made

import breeze.linalg
import breeze.numerics.abs
import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.linalg.{DenseVector, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.mllib
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Encoder}


trait LinearRegressionParams extends HasInputCol with HasOutputCol {
  def setInputCol(value: String): this.type = set(inputCol, value)
  def setOutputCol(value: String): this.type = set(outputCol, value)

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getInputCol, new VectorUDT())

    if (schema.fieldNames.contains($(outputCol))) {
      SchemaUtils.checkColumnType(schema, getOutputCol, new VectorUDT())
      schema
    } else {
      SchemaUtils.appendColumn(schema, schema(getInputCol).copy(name = getOutputCol))
    }
  }
}

class LinearRegression(override val uid: String) extends Estimator[LinearRegressionModel] with LinearRegressionParams
  with DefaultParamsWritable {
  val LR = 0.0001

  def this() = this(Identifiable.randomUID("linearRegression"))

  override def fit(dataset: Dataset[_]): LinearRegressionModel = {

    // Used to convert untyped dataframes to datasets with vectors
    implicit val encoder: Encoder[Vector] = ExpressionEncoder()

    val vectors: Dataset[Vector] = dataset.select(dataset($(inputCol)).as[Vector])

    val n_weights: Int = AttributeGroup.fromStructField((dataset.schema($(inputCol)))).numAttributes.getOrElse(vectors.first().size)
    val N: Long = vectors.rdd.count()

    val delta = 0.00001
    var weights: linalg.DenseVector[Double] = linalg.DenseVector.rand[Double](n_weights) // случайные начальные веса

    for (_ <- 1 to 100000) {
      var eps: Double = 0
      val stepWeights = vectors.rdd.mapPartitions((data: Iterator[Vector]) => {

        val summarizer = new MultivariateOnlineSummarizer()
        data.foreach(v => {
          val local_eps = (weights(0 until n_weights) *:* v.asBreeze(0 until n_weights)).reduce(_ + _) + weights(-1) - v.asBreeze(-1)
          eps = eps + local_eps

          var delta_weights =  v.asBreeze
          delta_weights(n_weights - 1) = 1
          delta_weights = delta_weights *:* linalg.DenseVector.fill(n_weights){local_eps*LR}
          summarizer.add(mllib.linalg.Vectors.fromBreeze(delta_weights))
        })
        Iterator(summarizer)
      }).reduce(_ merge _)


      if (abs(eps)/N < delta){
       // возвращем модель

        return copyValues(new LinearRegressionModel(Vectors.fromBreeze(weights))).setParent(this)
      } else {
        // делаем шаг
        weights = weights - stepWeights.mean.asBreeze
      }
    }
    copyValues(new LinearRegressionModel(Vectors.fromBreeze(weights))).setParent(this)

  }

  override def copy(extra: ParamMap): Estimator[LinearRegressionModel] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

}

object LinearRegression extends DefaultParamsReadable[LinearRegression]

class LinearRegressionModel private[made]( override val uid: String,
                                           val weights: DenseVector) extends Model[LinearRegressionModel] with LinearRegressionParams with MLWritable {


  private[made] def this(weights: Vector) =
    this(Identifiable.randomUID("linearRegression"), weights.toDense)

  override def copy(extra: ParamMap): LinearRegressionModel = copyValues(new LinearRegressionModel(weights))

  override def transform(dataset: Dataset[_]): DataFrame = {
    val transformUdf = dataset.sqlContext.udf.register(uid + "_transform",
      (x : Vector) => {
        Vectors.fromBreeze(x.asBreeze *:* weights.asBreeze(0 until x.size) + weights.asBreeze(x.size))
      })

    dataset.withColumn($(outputCol), transformUdf(dataset($(inputCol))))
  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  override def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)
      val vectors = Tuple1(weights.asInstanceOf[Vector])
      sqlContext.createDataFrame(Seq(vectors)).write.parquet(path + "/vectors")
    }
  }
}

object LinearRegressionModel extends MLReadable[LinearRegressionModel] {
  override def read: MLReader[LinearRegressionModel] = new MLReader[LinearRegressionModel] {
    override def load(path: String): LinearRegressionModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc)

      val vectors = sqlContext.read.parquet(path + "/vectors")

      // Used to convert untyped dataframes to datasets with vectors
      implicit val encoder: Encoder[Vector] = ExpressionEncoder()

      val weights = vectors.select(vectors("_1").as[Vector]).first()
      val model = new LinearRegressionModel(weights)
      metadata.getAndSetParams(model)
      model
    }
  }
}