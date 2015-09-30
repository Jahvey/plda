package com.tang

import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkContext, SparkConf}

import scala.StringBuilder
import scala.collection.mutable

/**
 * Created by Tang Lizhe on 2015/9/29.
 */
object TransToMpi {
  def main(args: Array[String]) {
    val minDocThreshold = args(0).toInt
    val hdfsHomeDir = "hdfs://10.107.20.25:9000/user/solo/"
    val inputPath = hdfsHomeDir + args(1)
    val outputPath = hdfsHomeDir + args(2)

    /**
     * If the word count value less than minDocFreq, the word will be filtered and will not be used
     * to calculate in following spark MLlib LDA computing.
     */
    val minDocFreq = minDocThreshold

    val conf = new SparkConf().setAppName("Spark Trans To MPI")
    val sc = new SparkContext(conf)

    /**
     * Let all documents from input file stored by doc:RDD[(Long, Array[String])].
     * Long position store docno.
     * Array[String] store all the words in document.
     */

    // Load and parse the data
    val file = sc.textFile(inputPath)
    val doc = file.map { line =>
      val w = line.trim.split(";;;;;", 2)
      val docno = w(0).trim.toLong
      val words = w(1).trim.split(" ")
      docno -> words
    }
    doc.persist(StorageLevel.MEMORY_AND_DISK)

    val totalWordCount = doc.map(_._2).flatMap(_.toIterator).map(w => (w, 1L)).reduceByKey(_ + _)
    totalWordCount.persist(StorageLevel.MEMORY_AND_DISK)
    //save to hdfs
    totalWordCount.saveAsTextFile(outputPath + "/TotalWordCount")

    val reducedWordCount = totalWordCount.filter(_._2 > minDocFreq)
    reducedWordCount.persist(StorageLevel.MEMORY_AND_DISK)
    //save to hdfs
    reducedWordCount.saveAsTextFile(outputPath + "/ReducedWordCount")

    val wordColumn = reducedWordCount.map(_._1).collect().toSet
    val bcWordColumn = sc.broadcast(wordColumn)

    val docCount = doc.mapValues { words =>
      val thisWordColumn = bcWordColumn.value
      val docMap = new mutable.HashMap[String, Int]()
      for (word <- words) {
        if (thisWordColumn.contains(word)) {
          docMap += word -> (docMap.getOrElse(word, 0) + 1)
        }
      }
      val docSB = new StringBuilder()
      for ((k, v) <- docMap) {
        docSB.append(k)
        docSB.append(" ")
        docSB.append(v)
        docSB.append(" ")
      }
      docSB.toString()
    }
    val mpiInput = docCount.map(_._2)
    val mpiInputNoBlankLine = mpiInput.filter(_.length > 0)

    mpiInput.cache()
    mpiInputNoBlankLine.cache()

    println("Total:" + mpiInput.count() + " NoBlank:" + mpiInputNoBlankLine.count())
    //save to hdfs
    mpiInputNoBlankLine.saveAsTextFile(outputPath + "/input")
  }
}
