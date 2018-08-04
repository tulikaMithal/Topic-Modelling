

import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions._
import org.apache.spark.ml.clustering.LDA
import org.apache.spark.ml.clustering.DistributedLDAModel


object Part1 {

  def main(args: Array[String]): Unit = {


    if (args.length == 0) {
      println("i need two parameters ")
    }

    val spark = SparkSession.builder.appName("Simple Application").getOrCreate()

    //Reading file data
    val reviewData = spark.read.option("header","true").json(args(0)).filter(col("text") isNotNull)

    var businessData = spark.read.option("header","true")
      .json(args(1)).withColumnRenamed("business_id","business_id2")
      .withColumnRenamed("stars","notusedstars").filter(col("state") === "AZ").filter(col("city") === "Phoenix")

    businessData = businessData.select("*").where(array_contains(col("categories"),"Pizza") && array_contains(col("categories"),"Italian") && array_contains(col("categories"),"Restaurants"))

    val arizonaData = businessData.join(reviewData, reviewData.col("business_id") === businessData.col("business_id2"),"left")

    var output=""

    output += "data count" + arizonaData.count() + "\n"



    //finding top and worst airline
    val top = arizonaData.groupBy("business_id").avg("stars").orderBy(desc("avg(stars)"))
    val worst = arizonaData.groupBy("business_id").avg("stars").orderBy(asc("avg(stars)"))


    output += "Top restaurant" + "\n" + top.collectAsList().get(0).mkString("\n") + "\n"
    output += "Worst restaurant" + "\n" + worst.collectAsList().get(0).mkString("\n") + "\n"

    val top_business = top.collectAsList().get(0).getString(0)
    val worst_business = worst.collectAsList().get(0).getString(0)



    //getting top and worst airline data
    val top_business_data = arizonaData.filter(col("business_id") === top_business)
    val worst_business_data = arizonaData.filter(col("business_id") === worst_business)

    val tokenizer = new Tokenizer()
      .setInputCol("text")
      .setOutputCol("words")

    val remover = new StopWordsRemover()
      .setInputCol("words")
      .setOutputCol("filtered")

    val vectorizer = new CountVectorizer()
      .setInputCol("filtered")
      .setOutputCol("features")
      .setVocabSize(2048)


    val lda = new LDA()
      .setK(2)
      .setMaxIter(50)
      .setOptimizer("em")



    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, remover, vectorizer, lda))

    val model_top= pipeline.fit(top_business_data)
    val model_worst = pipeline.fit(worst_business_data)



    val vectorizerModel_top = model_top.stages(2).asInstanceOf[CountVectorizerModel]

    val ldaModel_top = model_top.stages(3).asInstanceOf[DistributedLDAModel]

    val vocabList_top = vectorizerModel_top.vocabulary
    val termsIdx2Str_top = udf { (termIndices: Seq[Int]) => termIndices.map(idx => vocabList_top(idx)) }

    val topics = ldaModel_top.describeTopics(maxTermsPerTopic = 15)
      .withColumn("terms", termsIdx2Str_top(col("termIndices")))

    val row = topics.select("topic", "terms", "termWeights").collectAsList()

    output += "\n"+ row.toArray.mkString(" ")

    val vectorizerModel_worst = model_worst.stages(2).asInstanceOf[CountVectorizerModel]

    val ldaModel_worst = model_worst.stages(3).asInstanceOf[DistributedLDAModel]

    val vocabList_worst = vectorizerModel_worst.vocabulary
    val termsIdx2Str_worst = udf { (termIndices: Seq[Int]) => termIndices.map(idx => vocabList_worst(idx)) }

    // Review Results of LDA model with Online Variational Bayes
    val topics1 = ldaModel_worst.describeTopics(maxTermsPerTopic = 15)
      .withColumn("terms", termsIdx2Str_worst(col("termIndices")))



    val row1= topics1.select("topic", "terms", "termWeights").collectAsList()
    output += "\n"+ row1.toArray.mkString("\n")

    val sc = spark.sparkContext
    sc.parallelize(List(output)).saveAsTextFile(args(2))
    sc.stop()










  }

}


