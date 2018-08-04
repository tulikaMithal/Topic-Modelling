import org.apache.spark.ml.feature.{StopWordsRemover}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.{SparkSession}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions._
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.evaluation.{ MulticlassClassificationEvaluator}
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.sql.functions.{col}
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.mllib.evaluation.MulticlassMetrics


object Part2 {

  def main(args: Array[String]): Unit = {


    if (args.length == 0) {
      println("i need three parameters ")
    }

    val spark = SparkSession.builder.appName("Simple Application").getOrCreate()

    //Reading file data
    val reviewData = spark.read.option("header","true").json(args(0)).filter(col("text") isNotNull).select("text","stars","business_id").withColumnRenamed("stars","label")

    var businessData = spark.read.option("header","true")
      .json(args(1)).withColumnRenamed("business_id","business_id2")
      .withColumnRenamed("stars","notusedstars")

    businessData = businessData.select("business_id2").where(array_contains(col("categories"),"Restaurants"))

    val data = businessData.join(reviewData, reviewData.col("business_id") === businessData.col("business_id2"),"left").select("text","label","business_id").withColumnRenamed("business_id","id")

    var output = ""

    val Array(trainingData, testData) = data.randomSplit(Array(0.9, 0.1))

    val tokenizer = new Tokenizer()
      .setInputCol("text")
      .setOutputCol("words")

    val remover = new StopWordsRemover()
      .setInputCol("words")
      .setOutputCol("filtered")

    val hashingTF = new HashingTF()
      .setNumFeatures(10000)
      .setInputCol(remover.getOutputCol)
      .setOutputCol("rawFeatures")

    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")


    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.2)
      .setElasticNetParam(0.0)

    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.1, 0.01))
      .build()

    val pipeline = new Pipeline().setStages(Array(tokenizer, remover, hashingTF,idf))

    val model = pipeline.fit(trainingData)

    val preprocessedData  = model.transform(trainingData).select("features","label")

    val cv_lr = new CrossValidator()
      .setEstimator(lr)
      .setEstimatorParamMaps(paramGrid)
      .setEvaluator(new MulticlassClassificationEvaluator())
      .setNumFolds(5)



    val cv_lrModel = cv_lr.fit(preprocessedData)
    //val mlp_model = mlp.fit(preprocessedData)


    val modelTest = pipeline.fit(testData)

    val preprocessedTestData  = modelTest.transform(testData).select("label","features")


    val res = cv_lrModel.transform(preprocessedTestData).select("label","prediction")
    //val mlp_res =  mlp_model.transform(preprocessedTestData).select("label","prediction")


    import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
    val evaluator = new MulticlassClassificationEvaluator()
    evaluator.setLabelCol("label")
    evaluator.setMetricName("accuracy")


    val accuracy = evaluator.evaluate(res)


    //val accuracy1 = evaluator.evaluate(mlp_res)


    output += accuracy
    //output += accuracy1 +"\n" +precision1+"\n"+recall1

    val sc = spark.sparkContext
    sc.parallelize(List(output)).saveAsTextFile(args(2))
    sc.stop()

  }

}


