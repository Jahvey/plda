package com.tang

import org.apache.spark.mllib.clustering.{EMLDAOptimizer, LDA, LDAModel, DistributedLDAModel}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{RangePartitioner, HashPartitioner, SparkContext, SparkConf}


/**
 * Created by Tang Lizhe on 2015/9/20.
 */
object PldaFromMatrix {
  def main(args: Array[String]) {
    val numTopics = args(0).toInt
    val alpha = args(1).toDouble
    val beta = args(2).toDouble
    val maxIterations = args(3).toInt
    val minDocThreshold = args(4).toInt
    val hdfsHomeDir = "hdfs://10.107.20.25:9000/user/solo/"
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
    val docWordWithBlankLine = file.map { line =>
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

    //filter the empty document
    val filteredDocWord = docWordWithBlankLine.filter(_._2.numActives > 0)
    filteredDocWord.persist(StorageLevel.MEMORY_AND_DISK_2)
    //save to hdfs
    filteredDocWord.saveAsTextFile(outputPath + "/DocWordMatrix")

    /**
     * Modify the input data partition distribution
     */
    val docWord = filteredDocWord
    //val docWord = filteredDocWord.coalesce(9, true)
    //docWord.persist(StorageLevel.MEMORY_AND_DISK_2)

    var end = System.nanoTime()
    val timeDocWordConstruct = end - start

    /**
     * release all cached rdd
     */
    //filteredDocWord.unpersist()

    // Cluster the documents into three topics using LDA
    start = System.nanoTime()
    val ldaModel: LDAModel = new LDA()
      .setK(numTopics)
      .setAlpha(alpha)
      .setBeta(beta)
      .setMaxIterations(maxIterations)
      .setOptimizer(new EMLDAOptimizer)
      .run(docWord)
    end = System.nanoTime()
    val timeLdaRun = end - start

    /**
     * Save ldamodel to hdfs
     */
    ldaModel.save(sc, outputPath + "/LdaModel")

    /**
     * Load distributed lda model
     */
    val distributedLdaModel: DistributedLDAModel = DistributedLDAModel.load(sc, outputPath + "/LdaModel")
    val topDoc = distributedLdaModel.topicDistributions.top(1)
    val numDocuments = if (topDoc.size == 0) 0L else topDoc.apply(0)._1

    /**
     * Output the topic description by some important words.
     * This is unrecommended when the input document is very more. Because under this way, the word-topics matrix
     * will be very large. The matrix is represent in dense way, and the matrix can't be stored distributed.
     * It is just a matrix on single node, not a spark rdd.
     */
    start = System.nanoTime()
    val topicsMatrix = distributedLdaModel.topicsMatrix
    val rowArray = new Array[Vector](topicsMatrix.numRows)
    for (i <- Range(0, topicsMatrix.numRows)) {
      val colArray = new Array[Double](topicsMatrix.numCols)
      for (j <- Range(0, topicsMatrix.numCols))
        colArray(j) = topicsMatrix(i, j)
      rowArray(i) = Vectors.dense(colArray)
    }
    val describeTopicRDD = sc.makeRDD(rowArray, 18) //18 partition
    //save to hdfs
    describeTopicRDD.saveAsTextFile(outputPath + "/TopicDescription")

    val topDocumentPerTopic = distributedLdaModel.topDocumentsPerTopic(10)
    val topDocumentPerTopicSparse = topDocumentPerTopic.map { topic =>
      Vectors.sparse(numDocuments.toInt + 1, topic._1.map(_.toInt), topic._2) // docno start from 1
    }
    val topDocumentPerTopicRDD = sc.makeRDD[Vector](topDocumentPerTopicSparse, 1).zipWithIndex.map(_.swap)
    //save to hdfs
    topDocumentPerTopicRDD.saveAsTextFile(outputPath + "/TopDocumentPerTopic")
    end = System.nanoTime()
    val timeWordTopicMatrixOutput = end - start

    /**
     * Get the result of lda, document distributed by topic, result format
     * is example as: doc t1 t2 t3 ... tn, t is the score of each topic possess.
     */
    start = System.nanoTime()
    val topicDistribution = distributedLdaModel.topicDistributions.sortByKey()
    //save to hdfs
    topicDistribution.saveAsTextFile(outputPath + "/TopicDistribution")

    val topTopicPerDocument = distributedLdaModel.topTopicsPerDocument(10).map { doc =>
      (doc._1, Vectors.sparse(numTopics, doc._2, doc._3))
    }.sortByKey()
    //save to hdfs
    topTopicPerDocument.saveAsTextFile(outputPath + "/TopTopicPerDocument")
    end = System.nanoTime()
    val timeTopicDistributionOutput = end - start

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

  implicit val LongVectorOrdering = new Ordering[(Long, Vector)] {
    override def compare(x: (Long, Vector), y: (Long, Vector)): Int = {
      x._1.compareTo(y._1)
    }
  }
}
