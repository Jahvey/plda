package com.tang

import breeze.linalg.operators.DenseMatrixOps
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf

import org.apache.spark.mllib.clustering._
import org.apache.spark.mllib.linalg.{DenseVector, Vector, Vectors}
import org.apache.spark.storage.StorageLevel
import scala.collection.mutable

/**
 * Created by Tang Lizhe on 2015/9/11.
 */
object Plda {
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
    val doc = file.map { line =>
      val w = line.trim.split(";;;;;", 2)
      val docno = w(0).trim.toLong
      val words = w(1).trim.split(" ")
      docno -> words
    }
    doc.persist(StorageLevel.MEMORY_AND_DISK)
    /**
     * Get the max docno as numDocuments
     */
    val docTop = doc.top(1)
    val numDocuments = if (docTop.size == 0) 0L else docTop.apply(0)._1
    var end = System.nanoTime()
    val timeFileInput = end - start

    /**
     * To construct a word column, we use word count algorithm to produce a sorted
     * word array with count value, it's totalWordCount:RDD[(String, Long)].
     * To improve the performance of lda algorithm, we should reduce the size of word column by filtering
     * words which count value is less than minDocFreq, result is reducedWordCount:RDD[(String, Long)].
     */
    start = System.nanoTime()
    val totalWordCount = doc.map(_._2).flatMap(_.toIterator).map(w => (w, 1L)).reduceByKey(_ + _)
    totalWordCount.persist(StorageLevel.MEMORY_AND_DISK)
    //save to hdfs
    totalWordCount.saveAsTextFile(outputPath + "/TotalWordCount")

    val reducedWordCount = totalWordCount.filter(_._2 >= minDocFreq)
    reducedWordCount.persist(StorageLevel.MEMORY_AND_DISK)
    val numWords = reducedWordCount.count()
    //save to hdfs
    reducedWordCount.saveAsTextFile(outputPath + "/ReducedWordCount")
    end = System.nanoTime()
    val timeWordCount = end - start

    /**
     * The input format of MLlib lda is RDD[(Long, Vector)].
     * Long postion store the docno, Vector store the frequence of words in document distributed by
     * reduced column words.
     */
    start = System.nanoTime()
    val wordIndexMap = reducedWordCount.map(_._1).zipWithIndex().mapValues(_.toInt).collect().toMap
    val docWordWithBlankLine = doc.mapValues { words =>
      val docWordFreqMap = new mutable.HashMap[String, Int]()
      for (word <- words) {
        if (wordIndexMap.contains(word)) {
          docWordFreqMap += word -> (docWordFreqMap.getOrElse(word, 0) + 1)
        }
      }
      val docIndexFreqList = docWordFreqMap.map { p =>
        (wordIndexMap(p._1), p._2.toDouble)
      }.toArray.sorted
      Vectors.sparse(numWords.toInt, docIndexFreqList.map(_._1), docIndexFreqList.map(_._2))
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

    end = System.nanoTime()
    val timeDocWordConstruct = end - start

    /**
     * release all cached rdd
     */
    totalWordCount.unpersist()
    reducedWordCount.unpersist()
    //filteredDocWord.unpersist()
    doc.unpersist()

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

    /**
     * Output the topic description by some important words.
     * This is unrecommended when the input document is very more. Because under this way, the word-topics matrix
     * will be very large. Due to the matrix is represent in dense way, the matrix can't be stored distributed.
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
      (timeFileInput + timeWordCount + timeDocWordConstruct +
        timeLdaRun + timeTopicDistributionOutput + timeWordTopicMatrixOutput) / minute
    }")
    println("FileInput: " + timeFileInput / minute)
    println("WordCount: " + timeWordCount / minute)
    println("DocWordConstruct: " + timeDocWordConstruct / minute)
    println("LdaRun: " + timeLdaRun / minute)
    println("TopicDistributionOutput: " + timeTopicDistributionOutput / minute)
    println("WordTopicMatrixOutput: " + timeWordTopicMatrixOutput / minute)
    println("------------------------------------------------------------------------------------")
  }

  implicit val LongArrayOrdering = new Ordering[(Long, Array[String])] {
    override def compare(x: (Long, Array[String]), y: (Long, Array[String])): Int = {
      x._1.compareTo(y._1)
    }
  }

  implicit val LongVectorOrdering = new Ordering[(Long, Vector)] {
    override def compare(x: (Long, Vector), y: (Long, Vector)): Int = {
      x._1.compareTo(y._1)
    }
  }
}
