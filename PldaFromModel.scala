package com.tang

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.clustering._
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.storage.StorageLevel

import scala.collection.mutable

/**
 * Created by Tang Lizhe on 2015/9/11.
 */
object PldaFromModel {
  def main(args: Array[String]) {
    val hdfsHomeDir = "hdfs://10.107.20.25:9000/user/solo/"
    val inputPath = hdfsHomeDir + args(0)
    val outputPath = hdfsHomeDir + args(1)

    val conf = new SparkConf().setAppName("Spark LDA")
    val sc = new SparkContext(conf)

    /**
     * Load distributed lda model
     */
    val distributedLdaModel: DistributedLDAModel = DistributedLDAModel.load(sc, inputPath + "/LdaModel")
    val numWords = distributedLdaModel.vocabSize
    val topDoc = distributedLdaModel.topicDistributions.top(1)
    val numDocuments = if (topDoc.size == 0) 0L else topDoc.apply(0)._1
    
    /**
     * Output the topic description by some important words.
     * This is unrecommended when the input document is very more. Because under this way, the word-topics matrix
     * will be very large. The matrix is represent in dense way, and the matrix can't be stored distributed.
     * It is just a matrix on single node, not a spark rdd.
     */
    var start = System.nanoTime()
    val topicsMatrix = distributedLdaModel.topicsMatrix
    val numTopics = topicsMatrix.numCols
    val rowArray = new Array[Vector](topicsMatrix.numRows)
    for (i <- Range(0, topicsMatrix.numRows)) {
      val colArray = new Array[Double](topicsMatrix.numCols)
      for (j <- Range(0, topicsMatrix.numCols))
        colArray(j) = topicsMatrix(i, j)
      rowArray(i) = Vectors.dense(colArray)
    }
    val describeTopicRDD = sc.makeRDD(rowArray, 18)  //18 partition
    //save to hdfs
    describeTopicRDD.saveAsTextFile(outputPath + "/TopicDescription")

    val topDocumentPerTopic = distributedLdaModel.topDocumentsPerTopic(10)
    val topDocumentPerTopicSparse = topDocumentPerTopic.map { topic =>
      Vectors.sparse(numDocuments.toInt + 1, topic._1.map(_.toInt), topic._2) // docno start from 1
    }
    val topDocumentPerTopicRDD = sc.makeRDD[Vector](topDocumentPerTopicSparse, 1).zipWithIndex.map(_.swap)
    //save to hdfs
    topDocumentPerTopicRDD.saveAsTextFile(outputPath + "/TopDocumentPerTopic")
    var end = System.nanoTime()
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
      (timeTopicDistributionOutput + timeWordTopicMatrixOutput) / minute
    }")
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
