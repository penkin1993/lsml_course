package org.apache.spark.ml.made

import com.google.common.io.Files
import org.scalatest._
import flatspec._
import matchers._
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.DataFrame

class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {

  val delta = 0.0001
  lazy val data_train: DataFrame = LinearRegressionTest._data_train
  lazy val data_test: DataFrame = LinearRegressionTest._data_test
  lazy val vectors: Seq[Vector] = LinearRegressionTest._vectors_train

  "Model" should "predict input data" in {
    val model: LinearRegressionModel = new LinearRegressionModel(
      weights = Vectors.dense(2.0, -0.5).toDense,
    ).setInputCol("features")
      .setOutputCol("features")


    val vectors: Array[Vector] = model.transform(data_train).collect().map(_.getAs[Vector](0))

    vectors.length should be(1)

    vectors(0)(2) should be(12.0 +- delta)
    vectors(1)(2) should be(10.0 +- delta)
    vectors(2)(2) should be(2.0 +- delta)
    vectors(3)(2) should be(2.0 +- delta)
  }

  "Estimator" should "should produce functional model" in {
    val estimator = new LinearRegression()
      .setInputCol("features")
      .setOutputCol("features")

    val model = estimator.fit(data_train)

    validateModel(model, model.transform(data_test))
  }

  private def validateModel(model: LinearRegressionModel, data: DataFrame) = {

    val vectors: Array[Vector] = data.collect().map(_.getAs[Vector](0))

    vectors.length should be(4)
    model.weights.size should be(3)
  }

  "Estimator" should "work after re-read" in {

    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setInputCol("features")
        .setOutputCol("features")
    ))

    val tmpFolder = Files.createTempDir()

    pipeline.write.overwrite().save(tmpFolder.getAbsolutePath)

    val model = Pipeline.load(tmpFolder.getAbsolutePath).fit(data_train).stages(0).asInstanceOf[LinearRegressionModel]

  }

  "Model" should "work after re-read" in {

    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setInputCol("features")
        .setOutputCol("features")
    ))

    val model = pipeline.fit(data_train)

    val tmpFolder = Files.createTempDir()

    model.write.overwrite().save(tmpFolder.getAbsolutePath)

    val reRead = PipelineModel.load(tmpFolder.getAbsolutePath)

    validateModel(model.stages(0).asInstanceOf[LinearRegressionModel], reRead.transform(data_test))
  }
}

object LinearRegressionTest extends WithSpark {

  lazy val _vectors_train = Seq(
    Vectors.dense(13.5, 12, 12),
    Vectors.dense(-1, 0, 10),
    Vectors.dense(0, 2, 2),
    Vectors.dense(0, 2, 2))

  lazy val _data_train: DataFrame = {
    import sqlc.implicits._
    _vectors_train.map(x => Tuple1(x)).toDF("features")
  }

  lazy val _vectors_test = Seq(
    Vectors.dense(13.5, 12),
    Vectors.dense(-1, 0),
    Vectors.dense(0, 2),
    Vectors.dense(0, 2))

  lazy val _data_test: DataFrame = {
    import sqlc.implicits._
    _vectors_test.map(x => Tuple1(x)).toDF("features")
  }

}
