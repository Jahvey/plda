package com.tang

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf

import org.apache.spark.mllib.clustering.{LDA, DistributedLDAModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.{SparseMatrix, DenseMatrix}

import scala.io.Source

/**
 * Created by Tang Lizhe on 2015/9/12.
 * Just for testing some functions.
 */

object Test {
  def main(args: Array[String]): Unit = {
    val topicNum = args(0).toInt
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
    println("-------------------------------------------------------------------------------------------")
    println(docWord)
  }
}
