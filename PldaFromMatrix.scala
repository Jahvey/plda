package com.tang

import org.apache.spark.mllib.clustering.{EMLDAOptimizer, LDA, DistributedLDAModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkContext, SparkConf}


/**
 * Created by Tang Lizhe on 2015/9/20.
 */
object PldaFromMatrix {
  def main(args: Array[String]) {
    val topicNum = args(0).toInt
    val alpha = args(1).toDouble
    val beta = args(2).toDouble
    val maxIterations = args(3).toInt
    val minDocThreshold = args(4).toInt
    val hdfsHomeDir = "hdfs://node-25:9000/user/solo/"
    val inputPath = hdfsHomeDir + args(5)
    val outputPath = hdfsHomeDir + args(6)

    /**
     * If the word count value less than minDocFreq, the word will be filtered and will not be used
     * to calculate in following spark MLlib LDA computing.
     */
    val minDocFreq = minDocThreshold

    val conf = new SparkConf().setAppName("Spark LDA")
    val sc = new SparkContext(conf)

    /**
     * Let all documents from input file stored by doc:RDD[(Long, Array[String])].
     * Long position store docno.
     * Array[String] store all the words in document.
     */
    var start = System.nanoTime()
    // Load and parse the data
    val file = sc.textFile(inputPath)
    val docWord = file.map { line =>
      // space docno vector.size indexArray valueArray
      val w = line.trim.split("(\\()|(,\\()|(,\\[)|(\\],\\[)|(\\]\\)\\))")
      val docno = w(1).trim.toLong
      val length = w(2).trim.toInt
      var indexArray = Array[Int]()
      var valueArray = Array[Double]()
      if (w.length == 5) {
        indexArray = w(3).trim.split(",").map(_.toInt)
        valueArray = w(4).trim.split(",").map(_.toDouble)
      }
      val vector = Vectors.sparse(length, indexArray, valueArray)
      docno -> vector
    }
    docWord.persist(StorageLevel.MEMORY_AND_DISK_2)
    var end = System.nanoTime()
    val timeDocWordConstruct = end - start

    // Cluster the documents into three topics using LDA
    start = System.nanoTime()
    val distributedLdaModel: DistributedLDAModel = new LDA()
      .setK(topicNum)
      .setAlpha(alpha)
      .setBeta(beta)
      .setMaxIterations(maxIterations)
      .setOptimizer(new EMLDAOptimizer)
      .run(docWord).asInstanceOf[DistributedLDAModel]
    end = System.nanoTime()
    val timeLdaRun = end - start

    /**
     * Get the result of lda, document distributed by topic, result format
     * is example as: doc t1 t2 t3 ... tn, t is the score of each topic possess.
     */
    start = System.nanoTime()
    val topicDistribution = distributedLdaModel.topicDistributions
    //save to hdfs
    topicDistribution.saveAsTextFile(outputPath + "/TopicDistribution")
    end = System.nanoTime()
    val timeTopicDistributionOutput = end - start

    /**
     * Output the word-topics matrix.
     * This is unrecommended when the input document is very more. Because under this way, the word-topics matrix
     * will be very large. The matrix is represent in dense way, and the matrix can't be stored distributed.
     * It is just a matrix on single node, not a spark rdd.
     */
    start = System.nanoTime()
    /*    val wordTopicMatrix = distributedLdaModel.topicsMatrix
        val wordTopicArray = wordTopicMatrix.toString(wordTopicMatrix.numRows, wordTopicMatrix.numCols * 24).split("\n")
        val wordTopic = sc.parallelize(wordTopicArray, 2)
        //save to hdfs
        wordTopic.saveAsTextFile(outputPath + "/WordTopicMatrix")*/
    end = System.nanoTime()
    val timeWordTopicMatrixOutput = end - start

    //spark context stop
    sc.stop()


    // one minute = how much nano?
    val minute = (BigInt(10).pow(9) * 60).toDouble
    println("------------------------------------------------------------------------------------")
    println("Unit: Minute")
    println(s"Total time: ${
      (timeDocWordConstruct +
        timeLdaRun + timeTopicDistributionOutput + timeWordTopicMatrixOutput) / minute
    }")
    println("DocWordConstruct: " + timeDocWordConstruct / minute)
    println("LdaRun: " + timeLdaRun / minute)
    println("TopicDistributionOutput: " + timeTopicDistributionOutput / minute)
    println("WordTopicMatrixOutput: " + timeWordTopicMatrixOutput / minute)
    println("------------------------------------------------------------------------------------")
  }
}
