package com.tang

import org.apache.spark._
import org.apache.spark.mllib.clustering.{LDA, DistributedLDAModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.{SparseMatrix, DenseMatrix}

/**
 * Created by Tang Lizhe on 2015/9/12.
 * Just for testing some functions.
 */

object Test {
  def main(args: Array[String]): Unit = {
    val m = new DenseMatrix(2, 3, Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
    val cols = m.toString().split("\n")
    println(m.toString())

    //    val m2 = new SparseMatrix(2, 3, Array(0, 1, 2), Array(0, 1), Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
    //    println(m2)
  }
}
