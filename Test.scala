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
    val s2 = "(100,(5,[],[]))"

    val filename = "docWordMatrix10"
    val file = Source.fromFile(filename, "utf-8")
    val sList = file.getLines().toList
    val docWordList = sList.map { line =>
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
    docWordList foreach println
  }
}
