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
    val a = Array[(Int, Double)]((9, 10.0))
    val b = Vectors.sparse(10, a)
    println(b)
    println(b.numActives)
  }
}
